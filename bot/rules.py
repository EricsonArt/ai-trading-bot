"""Entry rules: the building blocks the strategy evolution combines.

A rule condition filters the model's buy signals, e.g. {"feature": "rsi", "op": "<", "value": 0.7}.
Ideas for rules come from two places:
  - mistakes: the bot compares its losing and winning signals and turns the clearest
    differences into candidate rules (mine_mistakes),
  - research: the LLM reads trading papers/news daily and proposes rules (research.py).
None of them is trusted on faith: evolution.py tries each idea on top of the current
champion and in new combinations, and only recipes that win on unseen data get deployed.
"""

import hashlib
import json
import time

import numpy as np

from . import config
from .features import RULE_FEATURES

PROPOSALS = config.DATA_DIR / "research" / "proposals.jsonl"
MISTAKES = config.DATA_DIR / "research" / "mistakes.json"
IDEA_DAYS = 14         # research ideas stay in the gene pool this long

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
    "rank_r96": "24h momentum rank among the coins (0 worst, 1 best)",
    "rank_r672": "7-day momentum rank among the coins (0 worst, 1 best)",
}
assert set(FEATURE_INFO) == set(RULE_FEATURES)


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


def mask(rules, df):
    keep = np.ones(len(df), dtype=bool)
    for r in rules:
        for c in r["conditions"]:
            x = df[c["feature"]].to_numpy()
            keep &= (x < c["value"]) if c["op"] == "<" else (x > c["value"])
    return keep


def blocking(rules, row):
    """Name of the first rule that blocks this live signal, or None."""
    for r in rules:
        for c in r["conditions"]:
            x = row.get(c["feature"])
            if x is None or not ((x < c["value"]) if c["op"] == "<" else (x > c["value"])):
                return r["name"]
    return None


def recent_ideas(days=IDEA_DAYS):
    """Research proposals from the last `days` days (the evolution's outside gene pool)."""
    if not PROPOSALS.exists():
        return []
    cutoff = time.time() * 1000 - days * 86_400_000
    out = {}
    for line in PROPOSALS.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("id") and r.get("conditions") and r.get("created_at", 0) >= cutoff:
            out[r["id"]] = {k: r.get(k, "") for k in ("id", "name", "why", "source", "conditions")}
    return list(out.values())


def mine_mistakes(df, net, threshold, discovery_end, model_version=None, write=True):
    """What did losing signals have in common? Look only at the older (search) period and
    turn the clearest differences into candidate rules for the evolution to test."""
    ts = df.ts.to_numpy()
    sig = (df.p.to_numpy() >= threshold) & (ts < discovery_end) & ~np.isnan(net)
    d, r = df[sig], net[sig]
    lose, win = r <= 0, r > 0
    summary = {"created_at": int(time.time() * 1000), "model_version": model_version,
               "signals": int(sig.sum()), "losers": int(lose.sum()), "patterns": [], "candidates": []}
    ideas = []
    if lose.sum() >= 20 and win.sum() >= 20:
        diffs = []
        for f in RULE_FEATURES:
            x = d[f].to_numpy(float)
            sd = float(x.std()) or 1.0
            diffs.append((abs(float(x[lose].mean() - x[win].mean())) / sd, f))
        diffs.sort(reverse=True)
        for strength, f in diffs[:5]:
            x = d[f].to_numpy(float)
            summary["patterns"].append({"feature": f, "meaning": FEATURE_INFO[f], "strength": round(strength, 3),
                                        "losers_avg": round(float(x[lose].mean()), 5),
                                        "winners_avg": round(float(x[win].mean()), 5)})
        base = float(r.mean())
        cands = []
        for _, f in diffs[:8]:
            x = d[f].to_numpy(float)
            for q in np.quantile(x, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]):
                for op in ("<", ">"):
                    kept = (x < q) if op == "<" else (x > q)
                    if kept.sum() >= len(x) // 2:
                        cands.append((float(r[kept].mean()) - base, f, op, float(q)))
        cands.sort(reverse=True)
        for gain, f, op, q in cands[:3]:
            if gain > 0.001:
                idea = clean({"name": f"unikaj {f} {'>=' if op == '<' else '<='} {q:.4g}",
                              "why": f"Stratne sygnały najbardziej różniły się tym: {FEATURE_INFO[f]}",
                              "conditions": [{"feature": f, "op": op, "value": q}]}, "own mistakes (auto)")
                ideas.append(idea)
                summary["candidates"].append({"name": idea["name"], "gain_on_old_data": round(gain, 5)})
    if write:
        MISTAKES.parent.mkdir(parents=True, exist_ok=True)
        MISTAKES.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return ideas
