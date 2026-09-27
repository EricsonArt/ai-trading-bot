import json

import numpy as np
import pandas as pd

from bot import config, evolution, research, rules
from bot.features import RULE_FEATURES

STEP = 15 * 60_000


def fake_oos(n=4000, seed=1):
    """Recorded signals where buying when overbought (rsi > 0.7) loses: a pattern worth learning."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(0, 1, n).astype("float32") for f in RULE_FEATURES})
    df["rsi"] = rng.uniform(0.2, 0.9, n).astype("float32")
    df["vol_96"] = np.float32(0.003)
    df["net"] = np.where(df.rsi > 0.7, -0.02, 0.01) + rng.normal(0, 0.004, n)
    df["ts"] = np.arange(n) * STEP * 2
    df["symbol"] = "BTC"
    df["exit_off"] = 1.0
    df["p"] = rng.uniform(0.55, 0.95, n)
    return df


def fake_bundle(version=1):
    oos = {s: fake_oos(seed=i) for i, s in enumerate(config.VARIANTS)}
    return {"oos": oos, "version": version, "variant": "trend",
            "metrics": {s: {"threshold": 0.6} for s in config.VARIANTS},
            "baselines": {s: {"search": -0.003, "select": -0.003, "holdout": -0.003} for s in config.VARIANTS},
            "feature_ranges": {f: [float(q) for q in oos["trend"][f].quantile([0.1, 0.5, 0.9])] for f in RULE_FEATURES}}


def recorded_exits(monkeypatch):
    # the fake history has no price candles: replay the recorded outcomes for every exit setting
    monkeypatch.setattr(evolution.Dreamer, "exits", lambda self, style, *a: (self.oos[style].net.to_numpy(),
                                                                          self.oos[style].exit_off.to_numpy()))


def test_clean_accepts_valid_and_refuses_junk():
    ok = rules.clean({"name": "not overbought", "conditions": [{"feature": "rsi", "op": "<", "value": 0.7}]}, "x")
    assert ok and ok["conditions"] == [{"feature": "rsi", "op": "<", "value": 0.7}] and len(ok["id"]) == 10
    assert rules.clean({"conditions": [{"feature": "moon_phase", "op": "<", "value": 1}]}, "x") is None
    assert rules.clean({"conditions": [{"feature": "rsi", "op": "=", "value": 1}]}, "x") is None
    assert rules.clean({"conditions": [{"feature": "rsi", "op": "<", "value": "nan"}]}, "x") is None
    assert rules.clean({"conditions": []}, "x") is None


def test_live_signal_is_blocked_by_rule():
    r = rules.clean({"name": "not overbought", "conditions": [{"feature": "rsi", "op": "<", "value": 0.7}]}, "t")
    assert rules.blocking([r], {"rsi": 0.8}) == "not overbought"
    assert rules.blocking([r], {"rsi": 0.5}) is None


def test_mistake_mining_finds_the_pattern():
    df = fake_oos()
    ideas = rules.mine_mistakes(df, df.net.to_numpy(), 0.6, discovery_end=df.ts.max() * 0.6)
    summary = json.loads(rules.MISTAKES.read_text(encoding="utf-8"))
    assert summary["patterns"][0]["feature"] == "rsi"
    assert any(c["feature"] == "rsi" and c["op"] == "<" for i in ideas for c in i["conditions"])


def test_evolution_discovers_rule_proves_edge_and_keeps_a_map(monkeypatch):
    recorded_exits(monkeypatch)
    champ = evolution.run(fake_bundle(), {}, [], budget_s=4, seed=3)
    g = champ["genome"]
    assert any(c["feature"] == "rsi" and c["op"] == "<" and c["value"] <= 0.8 for c in g["rules"])
    assert champ["edge"] and champ["stats"]["holdout"]["avg"] > 0.005
    world = json.loads(evolution.MAP.read_text(encoding="utf-8"))
    assert world["total_evaluated"] > 50 and world["deployments"][-1]["id"] == g["id"]
    archived = [json.loads(x) for x in evolution.ARCHIVE.read_text(encoding="utf-8").splitlines()]
    assert any(r["deployed"] for r in archived)
    # dreaming again can only replace the champion with something better on the newest data
    again = evolution.run(fake_bundle(version=2), {}, [], budget_s=2, seed=4)
    assert again["stats"]["holdout"]["score"] >= champ["stats"]["holdout"]["score"] - 1e-9


def test_research_ideas_enter_the_gene_pool(monkeypatch, tmp_path):
    recorded_exits(monkeypatch)
    for mod, attr, name in ((rules, "PROPOSALS", "proposals.jsonl"), (research, "DIR", ""),
                            (research, "DIGEST", "digest.json"), (research, "HISTORY", "digests.jsonl"),
                            (evolution, "IDEAS", "ideas.json"), (evolution, "CHAMPION", "champion.json"),
                            (evolution, "MAP", "map.json"), (evolution, "ARCHIVE", "archive.jsonl")):
        monkeypatch.setattr(mod, attr, tmp_path / name if name else tmp_path)
    monkeypatch.setattr(research, "gather", lambda: {"items": [{"kind": "news", "title": "t", "url": "u", "summary": ""}],
                                                     "fng": [], "used": ["news"], "failed": []})
    monkeypatch.setattr(research.brain, "ask", lambda *a, **k: {"digest": "calm day", "mood": "neutral", "ideas": [
        {"name": "skip overbought", "why": "article", "source": "news",
         "conditions": [{"feature": "rsi", "op": "<", "value": 0.7}]},
        {"name": "junk", "why": "", "source": "", "conditions": [{"feature": "nope", "op": "<", "value": 1}]}]})
    d = research.run("any-model", "local")
    assert len(d["ideas"]) == 1 and research.fresh(1)
    ideas = rules.recent_ideas()
    assert [i["name"] for i in ideas] == ["skip overbought"]
    evolution.run(fake_bundle(version=5), {}, ideas, budget_s=1, seed=5)
    verdict = json.loads(evolution.IDEAS.read_text(encoding="utf-8"))[ideas[0]["id"]]
    assert verdict["gain"] > 0  # the idea helps on data the search never used


def test_live_league_can_crown_a_challenger_and_block_live_losers(monkeypatch, tmp_path):
    recorded_exits(monkeypatch)
    for attr, name in (("CHAMPION", "c.json"), ("MAP", "m.json"), ("ARCHIVE", "a.jsonl"), ("IDEAS", "i.json")):
        monkeypatch.setattr(evolution, attr, tmp_path / name)
    first = evolution.run(fake_bundle(version=7), {}, [], budget_s=2, seed=7)
    assert first["challengers"]
    rival = first["challengers"][0]
    # live paper trading: the champion loses, a challenger wins clearly
    live = {first["genome"]["id"]: {"trades": 25, "avg": -0.01}, rival["id"]: {"trades": 25, "avg": 0.02}}
    second = evolution.run(fake_bundle(version=7), {}, [], budget_s=1, seed=8, live=live)
    assert second["genome"]["id"] == rival["id"] and second["reason"] == "won the live agent league"
    # a recipe that keeps losing live is never picked as the new champion
    third = evolution.run(fake_bundle(version=7), {}, [], budget_s=1, seed=9,
                          live={rival["id"]: {"trades": 30, "avg": -0.02}})
    assert rival["id"] not in [c["id"] for c in third["challengers"]]
