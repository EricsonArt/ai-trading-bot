"""The fast decision model ("reflex").

A gradient-boosted tree classifier estimates P(this trade ends in profit after fees).
Each retrain:
  1. labels every past candle with what the broker would really have earned,
  2. trains on older data, tests on the most recent ~20% it has never seen,
  3. picks the trading style + confidence threshold that made the most money on that
     unseen period (penalising lucky small samples),
  4. refits on everything so the live model knows the newest market behaviour.
If no style shows a real edge on unseen data, the bot keeps trading tiny (exploration).
"""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import config
from .features import FEATURES, RULE_FEATURES, add_labels, usable

MODEL_PATH = config.CACHE_DIR / "model.pkl"
STEP_MS = config.INTERVAL_MIN * 60_000


def _classifier(max_iter=400, early_stopping=True):
    return HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=150,
        l2_regularization=1.0, early_stopping=early_stopping, validation_fraction=0.1,
        n_iter_no_change=30, categorical_features=[FEATURES.index("sym")], random_state=7,
    )


def _thin(df):
    return df[(df.ts // STEP_MS) % config.TRAIN_STRIDE == 0]


def _weights(ts, ref_ts):
    age_days = (ref_ts - ts) / 86_400_000
    return 0.5 ** (age_days / config.RECENCY_HALF_LIFE_DAYS)


def simulate(symbols, ts, net, exit_off, proba, threshold):
    """Net returns of the trades the bot would take: one position per symbol at a time.

    Works on any subset of rows (e.g. only the signals), because "busy until" is tracked
    by time, not by row position."""
    out = []
    for s in np.unique(symbols):
        m = np.flatnonzero(symbols == s)
        m = m[np.argsort(ts[m], kind="stable")]
        free_at = -np.inf
        for i in m:
            if ts[i] >= free_at and proba[i] >= threshold:
                out.append(net[i])
                free_at = ts[i] + exit_off[i] * STEP_MS
    return np.array(out)


def simulate_df(df, proba, threshold):
    return simulate(df.symbol.to_numpy(), df.ts.to_numpy(), df.net.to_numpy(), df.exit_off.to_numpy(),
                    np.asarray(proba), threshold)


def trade_stats(rets):
    n = len(rets)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "avg": 0.0, "total": 0.0, "profit_factor": 0.0, "score": -1.0}
    wins, losses = rets[rets > 0], rets[rets <= 0]
    std = rets.std(ddof=1) if n > 1 else abs(rets[0])
    return {
        "trades": n,
        "win_rate": float(len(wins) / n),
        "avg": float(rets.mean()),
        "total": float(rets.sum()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else 99.0,
        # total profit minus one standard error: rewards edges that are not just luck
        "score": float(n * rets.mean() - np.sqrt(n) * std),
    }


def _evaluate_variant(frame, variant):
    lab = add_labels(frame, variant)
    lab = lab[usable(lab) & lab.resolved].sort_values("ts")
    H = config.VARIANTS[variant]["horizon"]
    t0, t1 = lab.ts.min(), lab.ts.max()
    val_start = t1 - (t1 - t0) * config.VALIDATION_FRACTION
    edges = np.linspace(val_start, t1 + 1, config.VALIDATION_FOLDS + 1)
    parts, probas, iters = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):  # walk forward: train on the past, test the next block
        train = _thin(lab[lab.ts < a - H * STEP_MS])  # purge: training labels must not peek into the test block
        test = lab[(lab.ts >= a) & (lab.ts < b)]
        clf = _classifier()
        clf.fit(train[FEATURES], train.y, sample_weight=_weights(train.ts.to_numpy(), a))
        parts.append(test)
        probas.append(clf.predict_proba(test[FEATURES])[:, 1])
        iters.append(clf.n_iter_)
    val = pd.concat(parts)
    proba = np.concatenate(probas)
    auc = float(roc_auc_score(val.y, proba)) if val.y.nunique() > 1 else 0.5
    best_t, best = None, None
    for t in config.THRESHOLDS:
        st = trade_stats(simulate_df(val, proba, t))
        if st["trades"] >= config.MIN_VALIDATION_TRADES and (best is None or st["score"] > best["score"]):
            best_t, best = t, st
    if best is None:  # too few signals anywhere: fall back to the loosest threshold
        best_t = config.THRESHOLDS[0]
        best = trade_stats(simulate_df(val, proba, best_t))
    # What would blindly buying every time have earned in the same period? The model only
    # has real skill if its picks beat this (otherwise it is just riding the market).
    naive = trade_stats(simulate_df(val, np.ones(len(val)), 0.0))
    best = {**best, "baseline_avg": naive["avg"], "baseline_trades": naive["trades"]}
    return {
        "variant": variant, "threshold": best_t, "auc": auc, "val": best,
        "base_rate": float(val.y.mean()), "n_iter": int(np.median(iters)),
        "val_rows": len(val), "labeled": lab, "val_df": val, "proba": proba,
    }


def train(frame, version=1):
    """Train every style, return the bundle: one final model per style + the out-of-sample
    evidence (walk-forward predictions) the strategy evolution replays."""
    started = time.time()
    results = [_evaluate_variant(frame, v) for v in config.VARIANTS]
    proven = [r for r in results if r["val"]["score"] > 0]
    # default style: a statistically proven one, otherwise the one that earned the most on unseen data
    best = max(proven, key=lambda r: r["val"]["score"]) if proven else max(results, key=lambda r: r["val"]["total"])
    clfs, data_until = {}, 0
    for r in results:
        lab = _thin(r["labeled"])
        data_until = max(data_until, int(lab.ts.max()))
        clf = _classifier(max_iter=max(r["n_iter"], 50), early_stopping=False)
        clf.fit(lab[FEATURES], lab.y, sample_weight=_weights(lab.ts.to_numpy(), lab.ts.max()))
        clfs[r["variant"]] = clf
    v = best["val"]
    edge = v["score"] > 0 and v["avg"] > 0 and v["avg"] > v["baseline_avg"] + 0.001
    return {
        "clfs": clfs,
        "symbols": list(config.SYMBOLS),
        "version": version,
        "variant": best["variant"],
        "threshold": best["threshold"],
        "edge": bool(edge),
        "trained_at": int(time.time() * 1000),
        "data_until": data_until,
        "train_seconds": round(time.time() - started, 1),
        "oos": {r["variant"]: _oos_signals(r) for r in results},
        "baselines": {r["variant"]: _baselines(r) for r in results},
        "feature_ranges": {f: [round(float(q), 5) for q in best["val_df"][f].quantile([0.1, 0.5, 0.9])]
                           for f in RULE_FEATURES},
        "metrics": {r["variant"]: {"auc": round(r["auc"], 4), "threshold": r["threshold"],
                                   "base_rate": round(r["base_rate"], 4), **{k: round(v, 5) for k, v in r["val"].items()}}
                    for r in results},
    }


def split_edges(ts):
    """Time boundaries of the evidence: older 60% = search, next 20% = select, newest 20% = holdout."""
    t0, t1 = float(np.min(ts)), float(np.max(ts))
    return t0 + (t1 - t0) * config.SPLITS[0], t0 + (t1 - t0) * config.SPLITS[1]


def _baselines(result):
    """Always-buy average per trade in each evidence period (what 'no skill' earns)."""
    val = result["val_df"]
    a, b = split_edges(val.ts.to_numpy())
    out = {}
    for name, part in (("search", val[val.ts < a]), ("select", val[(val.ts >= a) & (val.ts < b)]),
                       ("holdout", val[val.ts >= b])):
        out[name] = round(trade_stats(simulate_df(part, np.ones(len(part)), 0.0))["avg"], 5)
    return out


def _oos_signals(result):
    val = result["val_df"].assign(p=result["proba"])
    val = val[val.p >= config.EVOLUTION["min_threshold"]]
    cols = ["ts", "symbol", "net", "exit_off", "p"] + RULE_FEATURES
    out = val[cols].copy()
    out[RULE_FEATURES] = out[RULE_FEATURES].astype("float32")
    return out.sort_values("ts").reset_index(drop=True)


def save(bundle):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(bundle))
    tmp.replace(MODEL_PATH)


def load():
    if not MODEL_PATH.exists():
        return None
    try:
        return pickle.loads(MODEL_PATH.read_bytes())
    except Exception as exc:
        print(f"[model] could not load cached model ({exc}); will retrain")
        return None


def predict(bundle, rows, style=None):
    return bundle["clfs"][style or bundle["variant"]].predict_proba(rows[FEATURES])[:, 1]


def summary(bundle):
    """JSON-safe description for the state file and dashboard."""
    return {k: bundle[k] for k in ("version", "variant", "threshold", "edge", "trained_at",
                                   "data_until", "train_seconds", "metrics")}
