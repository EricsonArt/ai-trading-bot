"""Collect everything the dashboard shows into data/dashboard.json."""

import json
import time

from . import config
from .engine import (EQUITY, MODEL_HISTORY, PRACTICE_TRADES, STATE, TRADES, live_record, load_json, read_jsonl,
                     save_json, start_cash, trust)

OUT = config.DATA_DIR / "dashboard.json"
MAX_POINTS = 1500


def _equity_series(state):
    """[[ts, main, btc_benchmark, practice], ...] downsampled for the chart."""
    if not EQUITY.exists():
        return []
    rows = [r.split(",") for r in EQUITY.read_text(encoding="utf-8").strip().splitlines()[1:]]
    step = max(1, len(rows) // MAX_POINTS)
    picked = rows[::step] + ([rows[-1]] if rows and (len(rows) - 1) % step else [])
    bench0 = state.get("bench_start_px") or 1
    return [[int(r[0]), float(r[1]), round(start_cash(state) * float(r[3]) / bench0, 2), float(r[4])]
            for r in picked]


def _max_drawdown(values):
    peak, worst = float("-inf"), 0.0
    for v in values:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1)
    return worst


def _stats(trades):
    if not trades:
        return {"trades": 0}
    nets = [t["net"] for t in trades]
    wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    losses = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    return {
        "trades": len(trades), "win_rate": sum(n > 0 for n in nets) / len(nets),
        "avg_net": sum(nets) / len(nets), "pnl": round(sum(t["pnl"] for t in trades), 2),
        "profit_factor": round(wins / losses, 2) if losses > 0 else None,
        "best": max(nets), "worst": min(nets),
    }


def _account(acc, trades, prices, equity, week_ago, cash0):
    acct = acc.get("acct", {})
    positions = []
    for s, p in acct.get("positions", {}).items():
        px = prices.get(s, p["entry_px"])
        positions.append({"symbol": s, "entry_ts": p["entry_ts"], "entry_px": p["entry_px"], "price": px,
                          "value": round(p["qty"] * px, 2), "pnl_pct": px / p["entry_px"] - 1,
                          "tp_px": p["tp_px"], "sl_px": p["sl_px"], "held": p["held"], "horizon": p["horizon"]})
    return {
        "equity": round(equity, 2), "return": equity / cash0 - 1,
        "cash": round(acct.get("cash", equity), 2), "positions": positions,
        "pending": sorted(acct.get("pending", {})),
        "trades": trades[-40:][::-1],
        "stats": {"all": _stats(trades), "week": _stats([t for t in trades if t["exit_ts"] > week_ago])},
        "trust": {s: round(trust(v), 3) for s, v in acc.get("sym_perf", {}).items()},
        "benched_until": acc.get("benched_until", {}),
        "paused_until": acc.get("paused_until", 0),
    }


def _map_points(world):
    """Dots for the dream map: [generation, avg profit/trade on the 'select' period, went live?]."""
    pts = [[g["gen"], v, 0] for g in world.get("generations", [])[-60:] for v in g.get("sample", [])]
    path = config.DATA_DIR / "evolution" / "archive.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["deployed"]:
                pts.append([r["gen"], r["stats"]["select"]["avg"], 1])
    return pts


def _league(state):
    cash0 = start_cash(state)
    rows = []
    for rid, a in state.get("league", {}).items():
        eq = a.get("equity", cash0)
        rows.append({"id": rid, "role": a.get("role"), "recipe": a["recipe"], "joined": a["joined"],
                     "equity": round(eq, 2), "return": eq / cash0 - 1, **live_record(a),
                     "open": sorted(a["trader"]["acct"]["positions"]), "recent": a.get("recent_trades", [])[-5:][::-1]})
    return sorted(rows, key=lambda r: (r["role"] != "champion", -r["return"]))


def _readiness(state, main, benchmark_return):
    """When has the strategy earned a (small!) real-money test? All of these must hold."""
    start = state.get("main", {}).get("acct", {}).get("start_ts") or time.time() * 1000
    days = (time.time() * 1000 - start) / 86_400_000
    trades = main["stats"]["all"].get("trades", 0)
    strat = state.get("strategy") or {}
    checks = [  # shown on the (Polish) dashboard
        ("Strategia ma potwierdzoną przewagę na niewidzianej historii", bool(strat.get("edge")),
         "tak" if strat.get("edge") else "nie"),
        ("Konto główne handlowało na papierze 90+ dni", days >= 90, f"{days:.0f} dni"),
        ("Co najmniej 100 zamkniętych transakcji", trades >= 100, f"{trades} transakcji"),
        ("Zysk po opłatach", main["return"] > 0, f"{100 * main['return']:+.1f}%"),
        ("Lepiej niż samo trzymanie BTC", main["return"] > benchmark_return,
         f"{100 * (main['return'] - benchmark_return):+.1f} pkt"),
        ("Największy spadek mniejszy niż 20%", main["max_drawdown"] > -0.20, f"{100 * main['max_drawdown']:.1f}%"),
    ]
    return {"ready": all(ok for _, ok, _ in checks),
            "checks": [{"name": n, "ok": bool(ok), "value": v} for n, ok, v in checks]}


def _learning(state):
    """The evolution map, the champion recipe, mistakes found, ideas tested, today's reading."""
    world = load_json(config.DATA_DIR / "evolution" / "map.json", {}) or {}
    ideas = load_json(config.DATA_DIR / "evolution" / "ideas.json", {}) or {}
    return {
        "strategy": state.get("strategy"),
        "total_evaluated": world.get("total_evaluated", 0),
        "generations": world.get("generations", [])[-120:],
        "deployments": world.get("deployments", [])[-10:][::-1],
        "points": _map_points(world),
        "ideas": sorted(ideas.values(), key=lambda v: v["tested_at"], reverse=True)[:14],
        "mistakes": load_json(config.DATA_DIR / "research" / "mistakes.json", None),
        "research": load_json(config.DATA_DIR / "research" / "digest.json", None),
    }


def write():
    state = load_json(STATE, {})
    series = _equity_series(state)
    prices = state.get("prices", {})
    week_ago = time.time() * 1000 - 7 * 86_400_000
    cash0 = start_cash(state)
    last = series[-1] if series else [0, cash0, cash0, cash0]
    history = read_jsonl(MODEL_HISTORY)[-40:]
    brain = load_json(config.DATA_DIR / "brain.json", {})
    main = {**_account(state.get("main", {}), read_jsonl(TRADES), prices, last[1], week_ago, cash0),
            "max_drawdown": _max_drawdown([p[1] for p in series])}
    benchmark_return = last[2] / cash0 - 1
    save_json(OUT, {
        "generated_at": int(time.time() * 1000),
        "last_candle": state.get("last_ts"),
        "start_ts": state.get("main", {}).get("acct", {}).get("start_ts"),
        "start_cash": cash0,
        "benchmark_return": benchmark_return,
        "series": series,
        "main": main,
        "readiness": _readiness(state, main, benchmark_return),
        "league": _league(state),
        "practice": {**_account(state.get("practice", {}), read_jsonl(PRACTICE_TRADES), prices, last[3], week_ago, cash0),
                     "max_drawdown": _max_drawdown([p[3] for p in series])},
        "signals": state.get("main", {}).get("signals", {}),
        "prices": prices,
        "model": state.get("model"),
        "model_history": [{k: h.get(k) for k in ("version", "trained_at", "variant", "threshold", "edge")}
                          | {"avg": h["metrics"][h["variant"]]["avg"], "auc": h["metrics"][h["variant"]]["auc"],
                             "baseline_avg": h["metrics"][h["variant"]].get("baseline_avg")}
                          for h in history],
        "directives": state.get("directives"),
        "brain": {k: brain.get(k) for k in ("created_at", "source", "model", "directives", "reasoning",
                                            "new_lessons", "lessons", "seconds")} if brain else None,
        "brain_journal": read_jsonl(config.DATA_DIR / "brain_journal.jsonl")[-10:][::-1],
        "clm": load_json(config.DATA_DIR / "clm.json", None),
        "clm_test": load_json(config.DATA_DIR / "clm_test.json", None),
        "learning": _learning(state),
        "backtest": {k: v for k, v in (load_json(config.DATA_DIR / "backtest.json", {}) or {}).items()
                     if k != "trade_log"} or None,
        "config": {"symbols": config.SYMBOLS, "interval_min": config.INTERVAL_MIN, "fee": config.FEE,
                   "slippage": config.SLIPPAGE, "base_position": config.BASE_POSITION,
                   "retrain_hours": config.RETRAIN_HOURS},
    })
