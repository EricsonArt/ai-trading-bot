"""Real market candles from public exchange APIs (no account or key needed).

Primary source is Binance's public market-data mirror; Coinbase is the fallback.
Candles are cached in .cache/candles so each run only downloads what is new.
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import config

STEP_MS = config.INTERVAL_MIN * 60_000
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
_session = requests.Session()
_session.headers["User-Agent"] = "paper-trading-bot/1.0"


def now_ms() -> int:
    return int(time.time() * 1000)


def _get(url, params):
    for attempt in range(4):
        try:
            r = _session.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def _binance(symbol, start_ms, end_ms):
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get("https://data-api.binance.vision/api/v3/klines", {
            "symbol": f"{symbol}USDT", "interval": f"{config.INTERVAL_MIN}m",
            "startTime": cursor, "endTime": end_ms - 1, "limit": 1000,
        })
        if not batch:
            break
        rows += [[int(k[0]), *map(float, k[1:6])] for k in batch]
        cursor = int(batch[-1][0]) + STEP_MS
        if len(batch) < 1000:
            break
    return rows


def _coinbase(symbol, start_ms, end_ms):
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        stop = min(end_ms, cursor + 300 * STEP_MS)
        iso = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()
        batch = _get(f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles", {
            "granularity": config.INTERVAL_MIN * 60, "start": iso(cursor), "end": iso(stop - 1),
        })
        # Coinbase order: [time_s, low, high, open, close, volume], newest first
        rows += [[int(c[0]) * 1000, c[3], c[2], c[1], c[4], c[5]] for c in batch]
        cursor = stop
        time.sleep(0.15)
    return rows


def fetch(symbol, start_ms, end_ms):
    """Closed candles with open time in [start_ms, end_ms)."""
    try:
        rows = _binance(symbol, start_ms, end_ms)
    except Exception as exc:  # geo-block, outage, etc.
        print(f"[data] binance failed for {symbol} ({exc}); using coinbase")
        rows = _coinbase(symbol, start_ms, end_ms)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df[(df.ts >= start_ms) & (df.ts < end_ms)]


def _cache_path(symbol):
    return config.CACHE_DIR / "candles" / f"{symbol}.csv.gz"


def load(symbol) -> pd.DataFrame:
    path = _cache_path(symbol)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def regularize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per interval; missing candles become flat candles with zero volume."""
    df = df.drop_duplicates("ts", keep="last").sort_values("ts")
    if df.empty:
        return df.reset_index(drop=True)
    grid = np.arange(df.ts.iloc[0], df.ts.iloc[-1] + STEP_MS, STEP_MS, dtype=np.int64)
    df = df.set_index("ts").reindex(grid)
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df.rename_axis("ts").reset_index()


def update(symbol, until_ms=None) -> pd.DataFrame:
    """Bring the cached history up to the last fully closed candle and return it."""
    until_ms = until_ms or now_ms()
    last_closed_open = (until_ms // STEP_MS) * STEP_MS  # candles opening before this are closed
    oldest = last_closed_open - config.HISTORY_DAYS * 86_400_000
    df = load(symbol)
    if len(df) and df.ts.iloc[0] > oldest + STEP_MS:  # history window grew: backfill older candles
        df = pd.concat([fetch(symbol, oldest, int(df.ts.iloc[0])), df])
    start = int(df.ts.iloc[-1]) + STEP_MS if len(df) else oldest
    start = max(start, oldest)
    if start < last_closed_open:
        new = fetch(symbol, start, last_closed_open)
        df = pd.concat([df, new]) if len(df) else new
    df = regularize(df[df.ts >= oldest])
    path = _cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def update_all(until_ms=None) -> dict:
    return {s: update(s, until_ms) for s in config.SYMBOLS}
