"""Paper broker: fake money, real prices, realistic costs.

Orders decided at a candle's close fill at the next candle's open (+slippage, +fee).
Each open position has a take-profit, a stop-loss and a time limit. Exits are checked
on every candle's high/low; if both levels are touched in one candle we assume the stop
hit first (pessimistic). These rules mirror features.label_symbol exactly.
"""

from . import config


def new_account(start_ts, cash=None):
    return {
        "cash": float(cash or config.START_CASH),
        "positions": {},
        "pending": {},
        "start_ts": start_ts,
    }


def place(acct, symbol, amount, tp, sl, horizon, info):
    """Queue a buy for the next candle's open. tp/sl are fractions (0.02 = 2%)."""
    acct["pending"][symbol] = {"amount": round(amount, 2), "tp": tp, "sl": sl, "horizon": horizon, **info}


def fill_pending(acct, symbol, candle):
    order = acct["pending"].pop(symbol, None)
    if order is None or symbol in acct["positions"]:
        return None
    amount = min(order["amount"], acct["cash"])
    if amount < config.MIN_ORDER:
        return None
    px = candle["open"] * (1 + config.SLIPPAGE)
    qty = amount / (px * (1 + config.FEE))
    acct["cash"] -= amount
    pos = {
        "qty": qty, "entry_px": px, "cost": amount, "entry_ts": int(candle["ts"]),
        "tp_px": px * (1 + order["tp"]), "sl_px": px * (1 - order["sl"]),
        "horizon": order["horizon"], "held": 0,
        **{k: v for k, v in order.items() if k not in ("amount", "tp", "sl", "horizon")},
    }
    acct["positions"][symbol] = pos
    return pos


def check_exit(acct, symbol, candle):
    """Advance an open position by one candle; close it if a barrier or the time limit is hit."""
    pos = acct["positions"].get(symbol)
    if pos is None:
        return None
    pos["held"] += 1
    gap = pos["held"] > 1  # on the entry candle the open is our own fill, so no gap logic
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    if l <= pos["sl_px"]:
        price, reason = (o if gap and o <= pos["sl_px"] else pos["sl_px"]), "stop-loss"
    elif h >= pos["tp_px"]:
        price, reason = (o if gap and o >= pos["tp_px"] else pos["tp_px"]), "take-profit"
    elif pos["held"] >= pos["horizon"]:
        price, reason = c, "time-limit"
    else:
        return None
    return close(acct, symbol, price, int(candle["ts"]), reason)


def close(acct, symbol, price, ts, reason):
    pos = acct["positions"].pop(symbol)
    fill = price * (1 - config.SLIPPAGE)
    proceeds = pos["qty"] * fill * (1 - config.FEE)
    acct["cash"] += proceeds
    return {
        "symbol": symbol, "entry_ts": pos["entry_ts"], "exit_ts": ts,
        "entry_px": round(pos["entry_px"], 8), "exit_px": round(fill, 8),
        "cost": round(pos["cost"], 2), "proceeds": round(proceeds, 2),
        "pnl": round(proceeds - pos["cost"], 2), "net": round(proceeds / pos["cost"] - 1, 5),
        "reason": reason, "held": pos["held"],
        **{k: pos[k] for k in ("prob", "variant", "threshold", "mode", "ctx", "recipe") if k in pos},
    }


def equity(acct, prices):
    return acct["cash"] + sum(p["qty"] * prices[s] for s, p in acct["positions"].items())


def exposure(acct, prices):
    return (sum(p["qty"] * prices[s] for s, p in acct["positions"].items())
            + sum(o["amount"] for o in acct["pending"].values()))
