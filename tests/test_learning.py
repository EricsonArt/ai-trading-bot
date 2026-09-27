import json

import numpy as np
import pandas as pd

from bot import research, rules
from bot.features import RULE_FEATURES

STEP = 15 * 60_000


def fake_bundle(n=3000, seed=1, version=1):
    """Out-of-sample signals where high RSI means a losing trade (a pattern worth learning)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(0, 1, n).astype("float32") for f in RULE_FEATURES})
    df["rsi"] = rng.uniform(0.2, 0.9, n).astype("float32")
    df["net"] = np.where(df.rsi > 0.7, -0.02, 0.01) + rng.normal(0, 0.004, n)
    df["ts"] = np.arange(n) * STEP * 2
    df["symbol"] = "BTC"
    df["exit_off"] = 1.0
    df["p"] = 0.9
    return {"oos": df, "threshold": 0.5, "version": version, "variant": "trend",
            "metrics": {"trend": {"baseline_avg": -0.003}}}


def test_clean_accepts_valid_and_refuses_junk():
    ok = rules.clean({"name": "not overbought", "conditions": [{"feature": "rsi", "op": "<", "value": 0.7}]}, "x")
    assert ok and ok["conditions"] == [{"feature": "rsi", "op": "<", "value": 0.7}] and len(ok["id"]) == 10
    assert rules.clean({"conditions": [{"feature": "moon_phase", "op": "<", "value": 1}]}, "x") is None
    assert rules.clean({"conditions": [{"feature": "rsi", "op": "=", "value": 1}]}, "x") is None
    assert rules.clean({"conditions": [{"feature": "rsi", "op": "<", "value": "nan"}]}, "x") is None
    assert rules.clean({"conditions": []}, "x") is None


def test_judge_keeps_useful_rule_and_rejects_useless_one():
    b = fake_bundle()
    good = rules.clean({"conditions": [{"feature": "rsi", "op": "<", "value": 0.7}]}, "t")
    useless = rules.clean({"conditions": [{"feature": "vz", "op": "<", "value": 0.0}]}, "t")
    assert rules.judge(good, [], b)[0] is True
    assert rules.judge(useless, [], b)[0] is False


def test_live_signal_is_blocked_by_active_rule():
    r = rules.clean({"name": "not overbought", "conditions": [{"feature": "rsi", "op": "<", "value": 0.7}]}, "t")
    assert rules.blocking([r], {"rsi": 0.8}) == "not overbought"
    assert rules.blocking([r], {"rsi": 0.5}) is None


def test_mistake_mining_finds_the_pattern():
    mined = rules.mine_mistakes(fake_bundle())
    summary = json.loads(rules.MISTAKES.read_text(encoding="utf-8"))
    assert summary["patterns"][0]["feature"] == "rsi"
    assert any(c["feature"] == "rsi" and c["op"] == "<" for r in mined for c in r["conditions"])


def test_research_ideas_are_queued_then_judged(monkeypatch, tmp_path):
    monkeypatch.setattr(rules, "RULES", tmp_path / "rules.json")
    monkeypatch.setattr(rules, "PROPOSALS", tmp_path / "proposals.jsonl")
    monkeypatch.setattr(research, "DIR", tmp_path)
    monkeypatch.setattr(research, "DIGEST", tmp_path / "digest.json")
    monkeypatch.setattr(research, "HISTORY", tmp_path / "digests.jsonl")
    monkeypatch.setattr(research, "gather", lambda: {"items": [{"kind": "news", "title": "t", "url": "u", "summary": ""}],
                                                     "fng": [], "used": ["news"], "failed": []})
    monkeypatch.setattr(research.brain, "ask", lambda *a, **k: {"digest": "calm day", "mood": "neutral", "ideas": [
        {"name": "skip overbought", "why": "article", "source": "news",
         "conditions": [{"feature": "rsi", "op": "<", "value": 0.7}]},
        {"name": "junk", "why": "", "source": "", "conditions": [{"feature": "nope", "op": "<", "value": 1}]}]})
    d = research.run("any-model", "local")
    assert len(d["ideas"]) == 1 and research.fresh(1)
    book, _ = rules.review(fake_bundle())
    assert [r["status"] for r in book["rules"]] == ["active"]
    assert book["strategy"]["rules"] == 1
    # a new model whose evidence no longer supports the rule retires it
    flipped = fake_bundle(version=2)
    flipped["oos"]["net"] = -flipped["oos"]["net"]
    book, _ = rules.review(flipped)
    assert book["rules"][0]["status"] == "retired"
