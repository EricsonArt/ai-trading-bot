"""Replay the recent past exactly as the practice account (every model signal) would have traded it.

The model is retrained every `retrain_days` using only data available at that moment,
so the result is an honest estimate (no peeking at the future). Compared against simply
buying and holding BTC, and an equal mix of all coins, over the same period.
"""

import time

from . import brain, config, data, evolution, features, model
from .engine import DAY, Trader, plan, predict_rows, save_json


def run(days=180, retrain_days=30, evolve=True, evolve_seconds=90):
    started = time.time()
    candles = {s: data.load(s) for s in config.SYMBOLS}
    frame = features.build_frame(candles)
    t_end = min(int(df.ts.iloc[-1]) for df in candles.values())
    t_start = (t_end - days * DAY) // data.STEP_MS * data.STEP_MS
    closes = {s: df.set_index("ts").close for s, df in candles.items()}
    trader = Trader(Trader.new_state(t_start), practice=True)  # raw model: every signal is traded
    trades, series, blocks = [], [], []
    champ = None
    t = t_start
    while t <= t_end:
        block_end = min(t + retrain_days * DAY, t_end + data.STEP_MS)
        past = frame[frame.ts <= t - data.STEP_MS]
        bundle = model.train(past)
        if evolve:  # dream only on candles that existed at that moment
            champ = evolution.run(bundle, {s: df[df.ts < t] for s, df in candles.items()}, (),
                                  budget_s=evolve_seconds, persist=False, champion=champ)
            recipe, edge = champ["genome"], champ["edge"]
        else:
            recipe, edge = evolution.default_recipe(bundle, bundle["variant"]), bundle["edge"]
        blocks.append({"from": t, "recipe": evolution.describe(recipe), "edge": edge})
        print(f"[backtest] block from {time.strftime('%Y-%m-%d', time.gmtime(t / 1000))}: {evolution.describe(recipe)}")
        ts_list = list(range(t, block_end, data.STEP_MS))
        by_ts = predict_rows(frame, ts_list, bundle, recipe["style"])
        trade_plan = plan(recipe, edge)
        for ts in ts_list:
            if ts in by_ts:
                closed, eq = trader.step(ts, by_ts[ts], trade_plan, brain.DEFAULTS)
                trades += closed
                series.append((ts, eq))
        t = block_end

    eq_end = series[-1][1]
    peak, mdd = 0.0, 0.0
    for _, v in series:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    hold = {s: float(closes[s][t_end] / closes[s][t_start] - 1) for s in config.SYMBOLS}
    step = max(1, len(series) // 800)
    b0 = closes[config.BENCHMARK][t_start]
    result = {
        "created_at": int(time.time() * 1000), "days": days, "from": t_start, "to": t_end, "evolved": evolve,
        "final_equity": round(eq_end, 2), "return": eq_end / config.START_CASH - 1, "max_drawdown": mdd,
        "trades": len(trades), "win_rate": sum(x["net"] > 0 for x in trades) / len(trades) if trades else 0,
        "avg_trade": sum(x["net"] for x in trades) / len(trades) if trades else 0,
        "btc_hold": hold[config.BENCHMARK], "all_coins_hold": sum(hold.values()) / len(hold),
        "blocks": blocks,
        "trade_log": [{k: x[k] for k in ("symbol", "entry_ts", "exit_ts", "cost", "pnl", "net", "reason", "prob", "mode")}
                      for x in trades],
        "series": [[ts, round(v, 2), round(config.START_CASH * closes[config.BENCHMARK][ts] / b0, 2)]
                   for ts, v in series[::step]],
        "seconds": round(time.time() - started),
    }
    save_json(config.DATA_DIR / "backtest.json", result)
    print(f"[backtest] {days}d: ${config.START_CASH:.0f} -> ${eq_end:.2f} ({100 * result['return']:+.2f}%), "
          f"max drawdown {100 * mdd:.1f}%, {len(trades)} trades, win rate {100 * result['win_rate']:.0f}%, "
          f"BTC hold {100 * result['btc_hold']:+.1f}%, all-coins hold {100 * result['all_coins_hold']:+.1f}%")
    return result
