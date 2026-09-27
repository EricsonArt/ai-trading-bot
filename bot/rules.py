"""Learned trading rules: from the bot's own mistakes and from ideas found on the internet.

A rule is a filter on the model's buy signals, e.g. "only buy when rsi < 0.7".
Rules come from two places:
  - mistakes: after every retrain the bot compares its losing and winning out-of-sample
    signals and turns the clearest difference into candidate rules (mine_mistakes),
  - research: the LLM reads trading papers/news daily and proposes rules (research.py).
Nothing is trusted on faith. Every candidate is tested on the model's out-of-sample
signals from ~1.5 years of unseen data: it must raise the average profit per trade on
the whole period AND on the most recent third (which mined rules never saw). Active rules
are re-tested after every retrain and retired when they stop helping.
"""

import hashlib
import json
import time

import numpy as np

from . import config
from .features import RULE_FEATURES
from .model import simulate_df, trade_stats

RULES = config.DATA_DIR / "rules.json"
PROPOSALS = config.DATA_DIR / "research" / "proposals.jsonl"
MISTAKES = config.DATA_DIR / "research" / "mistakes.json"
MAX_ACTIVE = 5
MIN_GAIN = 0.001      # a rule must add at least +0.1% average profit per trade
MIN_TRADES = 40       # ...and leave enough trades to judge

FEATURE_INFO = {
    "r_1": "return over the last 15 minutes (0.01 = +1%)", "r_2": "return over the last 30 minutes",
    "r_4": "return over the last hour", "r_8": "return over the last 2 hours", "r_16": "return over the last 4 hours",
    "r_32": "return over the last 8 hours", "r_96": "return over the last 24 hours", "r_288": "return over the last 3 days",
    "r_672": "return over the last 7 days", "r_2016": "return over the last 21 days",
    "vol_16": "volatility of 15-min returns over the last 4 hours", "vol_96": "volatility of 15-min returns over 24 hours",
    "vol_ratio": "short vs long volatility (above 1 = volatility rising)",
    "atr": "average true range as a fraction of price", "rsi": "RSI(14) scaled 0..1 (0.7 overbought, 0.3 oversold)",
    "macd_hist": "MACD histogram / price (positive = upward momentum)",
    "bb": "position in Bollinger bands (+1 upper band, -1 lower band)",
    "d_ema50": "price vs its 50-candle (12.5 h) average", "d_ema200": "price vs its 200-candle (50 h) average",
    "ema_trend": "50-candle average vs 200-candle average (positive = uptrend)",
    "d_ema800": "price vs its ~8-day average", "dd_7d": "distance below the 7-day high (0 = at the high)",
    "vz": "trading volume vs 24h average (0.69 = normal, higher = busier)",
    "range_pos": "position inside the last 24h high-low range (0 = at the low, 1 = at the high)",
    "body": "last candle body / range (+1 strong green, -1 strong red)", "upper_wick": "upper wick / range",
    "lower_wick": "lower wick / range", "btc_r4": "Bitcoin return over the last hour",
    "btc_r16": "Bitcoin return over the last 4 hours", "btc_r96": "Bitcoin return over the last 24 hours",
    "btc_vol96": "Bitcoin 24h volatility", "rel_r96": "coin's 24h return minus Bitcoin's",
    "rel_r672": "coin's 7-day return minus Bitcoin's",
    "rank_r96": "24h momentum rank among the 6 coins (0 worst, 1 best)",
    "rank_r672": "7-day momentum rank among the 6 coins (0 worst, 1 best)",
}
assert set(FEATURE_INFO) == set(RULE_FEATURES)


# ---------- storage ----------

def load():
    try:
        return json.loads(RULES.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"rules": [], "strategy": None}


def save(book):
    RULES.parent.mkdir(parents=True, exist_ok=True)
    RULES.write_text(json.dumps(book, indent=1), encoding="utf-8")


def active(book=None):
    return [r for r in (book or load())["rules"] if r["status"] == "active"]


def rule_id(conditions):
    key = json.dumps(sorted((c["feature"], c["op"], round(float(c["value"]), 4)) for c in conditions))
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def clean(idea, source):
    """Turn a raw idea (from the LLM or the miner) into a valid rule, or None."""
    conds = []
    for c in (idea.get("conditions") or [])[:3]:
        try:
            f, op, v = c["feature"], c["op"], float(c["value"])
        except (KeyError, TypeError, ValueError):
            return None
        if f not in FEATURE_INFO or op not in ("<", ">") or not np.isfinite(v):
            return None
        conds.append({"feature": f, "op": op, "value": round(v, 5)})
    if not conds:
        return None
    return {"id": rule_id(conds), "name": str(idea.get("name", ""))[:80] or describe(conds),
            "why": str(idea.get("why", ""))[:300], "source": str(source)[:200], "conditions": conds}


def describe(conds):
    return " and ".join(f"{c['feature']} {c['op']} {c['value']:g}" for c in conds)


# ---------- applying ----------

def mask(rules, df):
    keep = np.ones(len(df), dtype=bool)
    for r in rules:
        for c in r["conditions"]:
            x = df[c["feature"]].to_numpy()
            keep &= (x < c["value"]) if c["op"] == "<" else (x > c["value"])
    return keep


def blocking(rules, row):
    """Name of the first active rule that blocks this live signal, or None."""
    for r in rules:
        for c in r["conditions"]:
            x = row.get(c["feature"])
            if x is None or not ((x < c["value"]) if c["op"] == "<" else (x > c["value"])):
                return r["name"]
    return None


# ---------- judging ----------

def evaluate(rules, oos, threshold):
    p = np.where(mask(rules, oos), oos.p.to_numpy(), 0.0)
    return trade_stats(simulate_df(oos, p, threshold))


def _recent(oos, frac=1 / 3):
    t0, t1 = oos.ts.min(), oos.ts.max()
    return oos[oos.ts >= t1 - (t1 - t0) * frac]


def judge(rule, current, bundle):
    """Does adding `rule` to the `current` rules improve results on unseen data?"""
    oos, thr = bundle["oos"], bundle["threshold"]
    rec = _recent(oos)
    before, after = evaluate(current, oos, thr), evaluate(current + [rule], oos, thr)
    before_r, after_r = evaluate(current, rec, thr), evaluate(current + [rule], rec, thr)
    ok = (after["trades"] >= MIN_TRADES and after["avg"] >= before["avg"] + MIN_GAIN
          and after["score"] > before["score"]
          and after_r["trades"] >= MIN_TRADES // 3 and after_r["avg"] >= before_r["avg"] + MIN_GAIN / 2)
    fmt = lambda s: {k: round(s[k], 5) for k in ("trades", "avg", "win_rate", "score")}
    return ok, {"before": fmt(before), "after": fmt(after), "recent_before": fmt(before_r),
                "recent_after": fmt(after_r), "gain": round(after["avg"] - before["avg"], 5)}


def strategy(bundle, book):
    """Model + active rules, judged on unseen data. Its edge gates the main account."""
    rules = active(book)
    st = evaluate(rules, bundle["oos"], bundle["threshold"])
    base = bundle["metrics"][bundle["variant"]]["baseline_avg"]
    edge = st["score"] > 0 and st["avg"] > 0 and st["avg"] > base + 0.001
    return {"edge": bool(edge), "rules": len(rules), "baseline_avg": base,
            **{k: round(st[k], 5) for k in ("trades", "avg", "win_rate", "score")}}


def review(bundle, new_ideas=()):
    """Re-test active rules on the current model's evidence, test new ideas, update rules.json."""
    book = load()
    now = int(time.time() * 1000)
    known = {r["id"] for r in book["rules"]}
    changed = False
    # 1) re-validate what is active whenever the model (and so its evidence) changed
    kept = []
    for r in active(book):
        if r.get("model_version") == bundle["version"]:
            kept.append(r)
            continue
        ok, test = judge(r, kept, bundle)
        r.update(test=test, tested_at=now, model_version=bundle["version"])
        if ok:
            kept.append(r)
        else:
            r["status"] = "retired"
            print(f"[rules] retired: {r['name']} (no longer helps)")
        changed = True
    # 2) new ideas: from mistakes and from research
    for idea in list(new_ideas) + _pending_proposals(known):
        rule = idea if "id" in idea else None
        if rule is None or rule["id"] in known:
            continue
        known.add(rule["id"])
        ok, test = judge(rule, kept, bundle)
        rule.update(test=test, tested_at=now, model_version=bundle["version"],
                    status="active" if ok and len(kept) < MAX_ACTIVE else "rejected")
        if rule["status"] == "active":
            kept.append(rule)
        print(f"[rules] {rule['status']}: {rule['name']} ({rule['source']}) gain {100 * test['gain']:+.2f}%/trade")
        book["rules"].append(rule)
        changed = True
    book["rules"] = book["rules"][-200:]
    book["strategy"] = strategy(bundle, book)
    book["strategy"]["model_version"] = bundle["version"]
    save(book)
    return book, changed


def _pending_proposals(known):
    if not PROPOSALS.exists():
        return []
    out = []
    for line in PROPOSALS.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("id") and r["id"] not in known and r.get("conditions"):
            out.append({k: r[k] for k in ("id", "name", "why", "source", "conditions") if k in r})
    return out


# ---------- learning from mistakes ----------

def mine_mistakes(bundle):
    """Compare losing vs winning signals on the older 2/3 of unseen data; propose filters.

    The newest third is kept out, so the judge can confirm on data the miner never saw."""
    oos, thr = bundle["oos"], bundle["threshold"]
    t0, t1 = oos.ts.min(), oos.ts.max()
    disc = oos[oos.ts < t1 - (t1 - t0) / 3]
    sig = disc[disc.p >= thr]
    losers, winners = sig[sig.net <= 0], sig[sig.net > 0]
    summary = {"created_at": int(time.time() * 1000), "model_version": bundle["version"],
               "signals": len(sig), "losers": len(losers), "patterns": [], "candidates": []}
    if len(losers) < 20 or len(winners) < 20:
        MISTAKES.parent.mkdir(parents=True, exist_ok=True)
        MISTAKES.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        return []
    # what did losing trades have in common? (standardised difference of means)
    diffs = []
    for f in RULE_FEATURES:
        sd = float(sig[f].std()) or 1.0
        diffs.append((abs(float(losers[f].mean() - winners[f].mean())) / sd, f))
    for d, f in sorted(diffs, reverse=True)[:5]:
        summary["patterns"].append({"feature": f, "meaning": FEATURE_INFO[f], "strength": round(d, 3),
                                    "losers_avg": round(float(losers[f].mean()), 5),
                                    "winners_avg": round(float(winners[f].mean()), 5)})
    # turn the strongest patterns into filters and keep the ones that help on the discovery data
    base = evaluate([], disc, thr)
    cands = []
    for _, f in sorted(diffs, reverse=True)[:8]:
        for q in np.quantile(sig[f], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]):
            for op in ("<", ">"):
                rule = clean({"name": f"avoid {f} {'>=' if op == '<' else '<='} {q:.4g}",
                              "why": f"Losing signals differed most in: {FEATURE_INFO[f]}",
                              "conditions": [{"feature": f, "op": op, "value": float(q)}]}, "own mistakes (auto)")
                st = evaluate([rule], disc, thr)
                if st["trades"] >= max(MIN_TRADES, base["trades"] // 2):
                    cands.append((st["avg"] - base["avg"], rule))
    cands.sort(key=lambda x: x[0], reverse=True)
    best = [r for gain, r in cands[:3] if gain > MIN_GAIN]
    summary["candidates"] = [{"name": r["name"], "conditions": r["conditions"]} for r in best]
    MISTAKES.parent.mkdir(parents=True, exist_ok=True)
    MISTAKES.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return best
