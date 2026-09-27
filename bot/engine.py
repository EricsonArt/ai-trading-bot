"""One bot cycle, plus the decision logic shared by live trading and backtests.

Two paper accounts run side by side on the same signals:
  - "main" only trades while the model has a proven edge on unseen data,
  - "practice" trades every signal, so the model's real skill is measured live
    without burning the main account.

Learning loops:
  1. the model retrains every few hours on the newest real outcomes (model.py),
  2. every symbol earns or loses "trust" from the bot's own closed trades: losing
     symbols get smaller positions and are benched for a day if they keep losing,
  3. the LLM brain reviews results and sets bounded risk directives (brain.py).
"""

import json
import time
from collections import deque

import pandas as pd

from . import brain, broker, config, data, features, model

STEP = data.STEP_MS
DAY = 86_400_000
STATE = config.DATA_DIR / "state.json"
TRADES = config.DATA_DIR / "trades.jsonl"
PRACTICE_TRADES = config.DATA_DIR / "practice_trades.jsonl"
EQUITY = config.DATA_DIR / "equity.csv"
MODEL_HISTORY = config.DATA_DIR / "model_history.jsonl"
LIVE_TAIL = 5000  # candles per symbol needed to compute current features (21-day lags + EMA warm-up)


# ---------- small file helpers ----------

def load_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, rows):
    if rows:
        with path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- decision logic ----------

def trust(recent):
    """1.0 = neutral. Built from the profit factor of the symbol's last trades, shrunk when few."""
    n = len(recent)
    if n == 0:
        return 1.0
    wins = sum(r for r in recent if r > 0)
    losses = -sum(r for r in recent if r <= 0)
    pf = 1.5 if losses == 0 else min(1.5, wins / losses)
    return 1 + (pf - 1) * n / (n + 10)


class Trader:
    """Portfolio + risk rules + the bot's memory of its own results."""

    def __init__(self, state, practice=False):
        self.s = state
        self.practice = practice
        self.recent_eq = deque(state.get("recent_equity", []), maxlen=97)

    @staticmethod
    def new_state(start_ts):
        return {
            "acct": broker.new_account(start_ts), "last_ts": start_ts - STEP,
            "peak": config.START_CASH, "paused_until": 0,
            "sym_perf": {}, "benched_until": {}, "signals": {}, "recent_equity": [],
            "counters": {"trades": 0, "wins": 0},
        }

    def step(self, t, rows, bundle, directives):
        """Process one candle close for every symbol. rows: {symbol: dict with prices, vol_96, p}."""
        acct, closed = self.s["acct"], []
        for sym, row in rows.items():
            broker.fill_pending(acct, sym, row)
            trade = broker.check_exit(acct, sym, row)
            if trade:
                closed.append(trade)
                self._learn_from(trade, t)
        prices = {sym: row["close"] for sym, row in rows.items()}
        for sym in acct["positions"]:
            prices.setdefault(sym, acct["positions"][sym]["entry_px"])
        eq = broker.equity(acct, prices)
        self.s["peak"] = max(self.s["peak"], eq)
        self.recent_eq.append((t, eq))
        day_ago = self.recent_eq[0][1]
        if len(self.recent_eq) == self.recent_eq.maxlen and eq < day_ago * (1 - config.DAILY_LOSS_PAUSE):
            self.s["paused_until"] = max(self.s["paused_until"], t + config.PAUSE_HOURS * 3_600_000)
        for sym, row in rows.items():
            self._decide(t, sym, row, eq, prices, bundle, directives)
        self.s["last_ts"] = t
        self.s["recent_equity"] = list(self.recent_eq)
        return closed, eq

    def _learn_from(self, trade, t):
        sym = trade["symbol"]
        perf = self.s["sym_perf"].setdefault(sym, [])
        perf.append(trade["net"])
        del perf[:-20]
        self.s["counters"]["trades"] += 1
        self.s["counters"]["wins"] += int(trade["net"] > 0)
        if len(perf) >= 5 and trust(perf) < config.SYMBOL_BENCH_TRUST:
            self.s["benched_until"][sym] = t + DAY  # keeps losing here: sit this symbol out for a day

    def _decide(self, t, sym, row, eq, prices, bundle, directives):
        acct = self.s["acct"]
        thr = round(bundle["threshold"] + directives["min_confidence_adj"], 4)
        sig = {"p": round(float(row["p"]), 4), "threshold": thr, "action": "wait", "ts": t}
        self.s["signals"][sym] = sig
        if sym in acct["positions"] or sym in acct["pending"]:
            sig["action"] = "holding"
            return
        if row["p"] < thr:
            return
        if not bundle["edge"] and not self.practice:
            sig["action"] = "wait: no proven edge yet (practice account takes it)"
            return
        bias = directives["symbol_bias"].get(sym, "normal")
        if t < self.s["paused_until"]:
            sig["action"] = "blocked: daily-loss pause"
            return
        if t < self.s["benched_until"].get(sym, 0):
            sig["action"] = "blocked: benched after losses"
            return
        if bias == "avoid":
            sig["action"] = "blocked: brain says avoid"
            return
        tr = trust(self.s["sym_perf"].get(sym, []))
        conf = min(1.5, max(0.75, 1 + (row["p"] - thr) * 4))
        size = eq * config.BASE_POSITION * directives["risk_level"] * tr * conf
        mode = "practice" if self.practice else "live"
        if eq < self.s["peak"] * (1 - config.DRAWDOWN_HALF_RISK):
            size *= 0.5
        if bias == "reduce":
            size *= 0.5
        pending_cash = sum(o["amount"] for o in acct["pending"].values())
        room = config.MAX_EXPOSURE * eq - broker.exposure(acct, prices)
        amount = min(size, acct["cash"] - pending_cash, room)
        if amount < config.MIN_ORDER:
            sig["action"] = "blocked: no free cash"
            return
        variant = bundle["variant"]
        tp, sl = features.barriers([row["vol_96"]], variant)
        broker.place(acct, sym, amount, float(tp[0]), float(sl[0]), config.VARIANTS[variant]["horizon"], {
            "prob": round(float(row["p"]), 4), "variant": variant, "threshold": thr, "mode": mode,
        })
        sig["action"] = f"buy ${amount:.0f} ({mode})"


# ---------- training ----------

def retrain(frame, cutoff, state):
    """Train on candles up to `cutoff` only, so replayed candles are never seen in advance."""
    version = state.get("model", {}).get("version", 0) + 1
    bundle = model.train(frame[frame.ts <= cutoff], version=version)
    bundle["features"] = list(features.FEATURES)
    model.save(bundle)
    (config.CACHE_DIR / "save-me").touch()  # tells the cloud job to persist the new model + price history
    summary = model.summary(bundle)
    state["model"] = summary
    append_jsonl(MODEL_HISTORY, [summary])
    print(f"[engine] model v{version}: style={bundle['variant']} threshold={bundle['threshold']} "
          f"edge={bundle['edge']} ({bundle['train_seconds']}s)")
    return bundle


def predict_rows(frame, ts_list, bundle):
    rows = frame[frame.ts.isin(ts_list) & features.usable(frame)].copy()
    rows["p"] = model.predict(bundle, rows) if len(rows) else []
    return {t: {r["symbol"]: r for r in g.to_dict("records")} for t, g in rows.groupby("ts")}


# ---------- live cycle ----------

def run(now_ms=None):
    started = time.time()
    candles = data.update_all(now_ms)
    t_max = min(int(df.ts.iloc[-1]) for df in candles.values())
    state = load_json(STATE)
    if state is None:
        state = {"last_ts": t_max - STEP, "bench_start_px": float(candles[config.BENCHMARK].close.iloc[-1]),
                 "main": Trader.new_state(t_max), "practice": Trader.new_state(t_max)}
        print(f"[engine] new paper accounts: ${config.START_CASH:.0f} main + ${config.START_CASH:.0f} practice")
    last = max(state["last_ts"], t_max - 7 * DAY)  # after a long outage replay at most a week
    ts_list = list(range(last + STEP, t_max + STEP, STEP))

    bundle = model.load()
    if bundle is None or bundle.get("features") != features.FEATURES:
        bundle = retrain(features.build_frame(candles), last, state)
    state["model"] = model.summary(bundle)

    live =features.build_frame({s: df.tail(LIVE_TAIL) for s, df in candles.items()})
    by_ts = predict_rows(live, ts_list, bundle)
    directives = brain.active_directives()
    main, practice = Trader(state["main"]), Trader(state["practice"], practice=True)
    closed, closed_practice, equity_rows = [], [], []
    for t in ts_list:
        if t not in by_ts:
            continue
        c, eq = main.step(t, by_ts[t], bundle, directives)
        cp, eq_p = practice.step(t, by_ts[t], bundle, brain.DEFAULTS)  # practice = the raw model, no brain
        closed += c
        closed_practice += cp
        equity_rows.append(f"{t},{eq:.2f},{state['main']['acct']['cash']:.2f},"
                           f"{by_ts[t][config.BENCHMARK]['close']},{eq_p:.2f}")
    state["last_ts"] = t_max
    state["prices"] = {s: float(df.close.iloc[-1]) for s, df in candles.items()}

    if (time.time() * 1000 - bundle["trained_at"]) > config.RETRAIN_HOURS * 3_600_000:
        retrain(features.build_frame(candles), t_max, state)

    append_jsonl(TRADES, closed)
    append_jsonl(PRACTICE_TRADES, closed_practice)
    if equity_rows:
        new_file = not EQUITY.exists()
        with EQUITY.open("a", encoding="utf-8") as f:
            if new_file:
                f.write("ts,equity,cash,btc,practice\n")
            f.write("\n".join(equity_rows) + "\n")
    state["updated_at"] = int(time.time() * 1000)
    state["directives"] = directives
    save_json(STATE, state)
    from . import report
    report.write()
    print(f"[engine] processed {len(ts_list)} candle(s), closed {len(closed)} main + {len(closed_practice)} practice "
          f"trade(s) in {time.time() - started:.1f}s")
    return state
