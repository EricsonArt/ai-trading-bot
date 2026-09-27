import numpy as np
import pytest

from bot import brain, broker, config, features
from bot.engine import Trader


def plan(edge=True, **kw):
    return {"id": "test", "style": "trend", "threshold": 0.5, "tp": 1.8, "sl": 0.9, "horizon": 64, "size": 0.25,
            "rules": [], "edge": edge, **kw}
from conftest import make_candles


def test_labels_match_broker(candles):
    """The training target must equal what the paper broker actually books."""
    f = features.indicators(candles)
    for variant in config.VARIANTS:
        net, off, resolved = features.label_symbol(f, variant)
        H = config.VARIANTS[variant]["horizon"]
        tp, sl = features.variant_barriers(f.vol_96, variant)
        checked = 0
        for i in range(300, len(f) - H - 1, 97):
            acct = broker.new_account(0)
            broker.place(acct, "X", 500.0, float(tp[i]), float(sl[i]), H, {})
            trade, j = None, 0
            while trade is None:
                j += 1
                row = f.iloc[i + j].to_dict()
                if j == 1:
                    broker.fill_pending(acct, "X", row)
                trade = broker.check_exit(acct, "X", row)
            assert resolved[i]
            assert off[i] == j, (variant, i)
            assert trade["net"] == pytest.approx(net[i], abs=2e-5), (variant, i)
            checked += 1
        assert checked > 20


def test_features_do_not_peek_into_the_future(candles):
    k = 3000
    base = features.indicators(candles)
    changed = candles.copy()
    changed.loc[k + 1:, ["open", "high", "low", "close"]] *= 1.5
    changed.loc[k + 1:, "volume"] *= 3
    alt = features.indicators(changed)
    cols = [c for c in features.FEATURES if c in base.columns]
    assert np.allclose(base.loc[:k, cols].to_numpy(), alt.loc[:k, cols].to_numpy(), equal_nan=True)


def test_round_trip_costs_fees_and_slippage():
    acct = broker.new_account(0)
    broker.place(acct, "X", 100.0, 0.5, 0.5, 10, {})
    broker.fill_pending(acct, "X", {"ts": 1, "open": 50.0, "high": 50, "low": 50, "close": 50})
    trade = broker.close(acct, "X", 50.0, 2, "manual")
    expected = (1 - config.SLIPPAGE) * (1 - config.FEE) / ((1 + config.SLIPPAGE) * (1 + config.FEE)) - 1
    assert trade["net"] == pytest.approx(expected, abs=1e-5)
    assert acct["cash"] == pytest.approx(config.START_CASH - 100 + trade["proceeds"], abs=0.01)


def test_brain_output_is_clamped():
    d = brain.sanitize({"risk_level": 7, "min_confidence_adj": -3, "avoid": ["BTC", "ETH", "SOL", "XRP", "NOPE"],
                        "reduce": ["DOGE"], "regime": "moon"})
    assert d["risk_level"] == 1.0 and d["min_confidence_adj"] == -0.02
    assert list(d["symbol_bias"].values()).count("avoid") == 3 and "NOPE" not in d["symbol_bias"]
    assert d["symbol_bias"]["DOGE"] == "reduce" and d["regime"] == "unknown"
    assert brain.sanitize({})["risk_level"] == brain.DEFAULTS["risk_level"]


def _rows(t, price, p):
    return {s: {"ts": t, "open": price, "high": price * 1.001, "low": price * 0.999, "close": price,
                "vol_96": 0.003, "p": p} for s in config.SYMBOLS}


def test_trader_never_exceeds_cash_or_exposure():
    trader = Trader(Trader.new_state(0))
    for k in range(40):
        trader.step(k * 900_000, _rows(k * 900_000, 100.0, 0.99), plan(), brain.DEFAULTS)
        acct = trader.s["acct"]
        assert acct["cash"] >= -1e-6
        prices = {s: 100.0 for s in config.SYMBOLS}
        assert broker.exposure(acct, prices) <= config.MAX_EXPOSURE * broker.equity(acct, prices) + 1e-6
    assert trader.s["acct"]["positions"]  # it did trade


def test_main_waits_without_edge_but_practice_trades():
    main, practice = Trader(Trader.new_state(0)), Trader(Trader.new_state(0), practice=True)
    main.step(0, _rows(0, 100.0, 0.99), plan(edge=False), brain.DEFAULTS)
    practice.step(0, _rows(0, 100.0, 0.99), plan(edge=False), brain.DEFAULTS)
    assert not main.s["acct"]["pending"] and main.s["signals"]["BTC"]["action"].startswith("wait")
    assert set(practice.s["acct"]["pending"]) and all(o["mode"] == "practice"
                                                      for o in practice.s["acct"]["pending"].values())


def test_symbol_gets_benched_after_repeated_losses():
    trader = Trader(Trader.new_state(0))
    for k in range(8):
        trader._learn_from({"symbol": "SOL", "net": -0.02}, t=k)
    assert trader.s["benched_until"]["SOL"] > 0
    trader.step(10, _rows(10, 100.0, 0.99), plan(), brain.DEFAULTS)
    assert "SOL" not in trader.s["acct"]["pending"]
    assert trader.s["signals"]["SOL"]["action"].startswith("blocked")


def test_full_cycle_offline(monkeypatch):
    """Two live cycles on synthetic markets: trains, trades, saves state and dashboard."""
    from bot import data, engine, report

    history = {s: make_candles(n=4200, seed=i) for i, s in enumerate(config.SYMBOLS)}
    feed = {"n": 4000}
    monkeypatch.setattr(data, "update_all", lambda now=None: {s: df.head(feed["n"]).copy() for s, df in history.items()})
    monkeypatch.setattr(config, "MIN_VALIDATION_TRADES", 5)
    monkeypatch.setitem(config.EVOLUTION, "budget_seconds", 3)
    monkeypatch.setitem(config.EVOLUTION, "min_trades", {"search": 5, "select": 2, "holdout": 2})
    s1 = engine.run()
    assert s1["model"]["version"] == 1 and s1["last_ts"] == history["BTC"].ts.iloc[3999]
    feed["n"] = 4200  # 200 new candles arrive: the bot replays all of them
    s2 = engine.run()
    assert s2["last_ts"] == history["BTC"].ts.iloc[4199]
    dash = engine.load_json(report.OUT)
    assert dash["main"]["equity"] > 0 and dash["practice"]["equity"] > 0 and len(dash["series"]) >= 200
    assert set(dash["signals"]) == set(config.SYMBOLS)
    assert brain.build_report()[0].count("account") >= 2  # the brain's report renders from the saved state
    assert dash["learning"]["strategy"]["description"] and dash["learning"]["total_evaluated"] > 10


def test_changing_the_fake_money_archives_old_results(monkeypatch):
    import pandas as pd
    from bot import data, engine
    engine.save_json(engine.STATE, {"start_cash": 1000.0, "model": {"version": 3}, "main": {}, "practice": {}})
    engine.append_jsonl(engine.TRADES, [{"symbol": "BTC", "net": 0.01}])
    monkeypatch.setattr(data, "fetch", lambda *a: pd.DataFrame({"ts": [1_000_000 * 900], "close": [50_000.0]}))
    engine.reset(5000)
    st = engine.load_json(engine.STATE)
    assert st["start_cash"] == 5000 and st["main"]["acct"]["cash"] == 5000 and st["practice"]["peak"] == 5000
    assert st["model"] == {"version": 3} and not engine.TRADES.exists()
    assert list((config.DATA_DIR / "archive").glob("*/trades.jsonl"))
    dash = engine.load_json(config.DATA_DIR / "dashboard.json")
    assert dash["start_cash"] == 5000 and dash["main"]["equity"] == 5000
