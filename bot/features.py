"""Turn raw candles into model inputs (features) and training targets (labels).

Every feature at candle t uses only candles <= t, so the model never sees the future.
Labels replay the exact exit rules the paper broker uses (take-profit, stop-loss,
time limit, fees and slippage), so the model learns what the broker will actually earn.
"""

import numpy as np
import pandas as pd

from . import config

RETURN_LAGS = [1, 2, 4, 8, 16, 32, 96, 288, 672, 2016]  # up to 21 days
FEATURES = (
    [f"r_{k}" for k in RETURN_LAGS]
    + ["vol_16", "vol_96", "vol_ratio", "atr", "rsi", "macd_hist", "bb", "d_ema50", "d_ema200",
       "ema_trend", "d_ema800", "dd_7d", "vz", "range_pos", "body", "upper_wick", "lower_wick",
       "hour_sin", "hour_cos", "dow_sin", "dow_cos",
       "btc_r4", "btc_r16", "btc_r96", "btc_vol96", "rel_r96", "rel_r672", "rank_r96", "rank_r672", "sym"]
)


# Features that learned rules may use (numeric, human-readable; no time-of-day or coin id).
RULE_FEATURES = [f for f in FEATURES if f not in ("sym", "hour_sin", "hour_cos", "dow_sin", "dow_cos")]


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    lr = np.log(c).diff()
    out = pd.DataFrame({"ts": df.ts.values, "open": o, "high": h, "low": l, "close": c})
    for k in RETURN_LAGS:
        out[f"r_{k}"] = np.log(c / c.shift(k))
    out["vol_16"] = lr.rolling(16).std()
    out["vol_96"] = lr.rolling(96).std()
    out["vol_ratio"] = out.vol_16 / out.vol_96
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = (gain / (gain + loss)).fillna(0.5)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb"] = (c - sma20) / (2 * std20)
    ema50, ema200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    out["d_ema50"] = c / ema50 - 1
    out["d_ema200"] = c / ema200 - 1
    out["ema_trend"] = ema50 / ema200 - 1
    out["d_ema800"] = c / c.ewm(span=800, adjust=False).mean() - 1  # ~200-hour trend
    out["dd_7d"] = c / h.rolling(672).max() - 1                        # distance from 7-day high
    out["vz"] = np.log1p(v / v.rolling(96).mean())
    hi96, lo96 = h.rolling(96).max(), l.rolling(96).min()
    out["range_pos"] = (c - lo96) / (hi96 - lo96)
    rng = (h - l).replace(0, np.nan)
    out["body"] = ((c - o) / rng).fillna(0)
    out["upper_wick"] = ((h - np.maximum(o, c)) / rng).fillna(0)
    out["lower_wick"] = ((np.minimum(o, c) - l) / rng).fillna(0)
    dt = pd.to_datetime(out.ts, unit="ms", utc=True)
    hour = dt.dt.hour + dt.dt.minute / 60
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7), np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[out.index[: config.WARMUP_CANDLES], "r_1"] = np.nan  # marks warm-up rows as unusable
    return out


def build_frame(candles: dict) -> pd.DataFrame:
    """All symbols stacked, with BTC's moves attached as market context."""
    frames = {s: indicators(df) for s, df in candles.items() if len(df)}
    btc = frames[config.BENCHMARK][["ts", "r_4", "r_16", "r_96", "r_672", "vol_96"]]
    btc = btc.rename(columns={"r_4": "btc_r4", "r_16": "btc_r16", "r_96": "btc_r96", "r_672": "btc_r672",
                              "vol_96": "btc_vol96"})
    parts = []
    for s, f in frames.items():
        f = f.merge(btc, on="ts", how="left")
        f["rel_r96"] = f.r_96 - f.btc_r96      # strength relative to BTC
        f["rel_r672"] = f.r_672 - f.btc_r672
        f.insert(1, "symbol", s)
        f["sym"] = config.SYMBOLS.index(s)
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    # cross-sectional momentum: how this coin ranks against the others right now (0 = worst, 1 = best)
    for k in ("r_96", "r_672"):
        out["rank_" + k.replace("_", "")] = out.groupby("ts")[k].rank(pct=True)
    return out


def barriers(vol_96, variant):
    """Take-profit / stop-loss distances (as fractions) for a given style."""
    v = config.VARIANTS[variant]
    move = np.asarray(vol_96, dtype=float) * np.sqrt(v["horizon"])
    return np.maximum(config.MIN_TP, v["tp"] * move), np.maximum(config.MIN_SL, v["sl"] * move)


def net_return(entry_fill, exit_price):
    """Return after fees on both sides; entry_fill already includes slippage."""
    exit_fill = exit_price * (1 - config.SLIPPAGE)
    return exit_fill * (1 - config.FEE) / (entry_fill * (1 + config.FEE)) - 1


def label_symbol(f: pd.DataFrame, variant: str):
    """Triple-barrier outcome of buying at the next candle's open, for every row of one symbol.

    Returns (net_return, exit_offset, resolved). exit_offset counts candles after the
    decision candle (1 = exited during the entry candle).
    """
    H = config.VARIANTS[variant]["horizon"]
    o, h, l, c = (f[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(f)
    tp, sl = barriers(f.vol_96.to_numpy(float), variant)
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:] * (1 + config.SLIPPAGE)
    tp_px, sl_px = entry * (1 + tp), entry * (1 - sl)
    exit_px = np.full(n, np.nan)
    exit_off = np.full(n, np.nan)
    idx = np.arange(n)
    for j in range(1, H + 1):
        k = idx + j
        ok = (k < n) & np.isnan(exit_px) & ~np.isnan(entry)
        kk = np.where(ok, k, 0)
        hit_sl = ok & (l[kk] <= sl_px)
        hit_tp = ok & ~hit_sl & (h[kk] >= tp_px)
        gap = j > 1
        sl_fill = np.where(gap & (o[kk] <= sl_px), o[kk], sl_px)
        tp_fill = np.where(gap & (o[kk] >= tp_px), o[kk], tp_px)
        timeout = ok & ~hit_sl & ~hit_tp & (j == H)
        exit_px = np.where(hit_sl, sl_fill, np.where(hit_tp, tp_fill, np.where(timeout, c[kk], exit_px)))
        exit_off = np.where(hit_sl | hit_tp | timeout, j, exit_off)
    resolved = idx + H < n
    return net_return(entry, exit_px), exit_off, resolved


def add_labels(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["net"], frame["exit_off"], frame["resolved"] = np.nan, np.nan, False
    for s, f in frame.groupby("symbol", sort=False):
        net, off, res = label_symbol(f, variant)
        frame.loc[f.index, "net"] = net
        frame.loc[f.index, "exit_off"] = off
        frame.loc[f.index, "resolved"] = res
    frame["y"] = (frame.net > 0).astype(int)
    return frame


def usable(frame: pd.DataFrame) -> pd.Series:
    return frame[FEATURES].notna().all(axis=1)
