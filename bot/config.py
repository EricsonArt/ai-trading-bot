"""All tunable settings in one place."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BOT_DATA_DIR", ROOT / "data"))      # committed: portfolio, trades, brain notes
CACHE_DIR = Path(os.environ.get("BOT_CACHE_DIR", ROOT / ".cache"))  # not committed: candles, trained model

# Market: crypto trades 24/7, so the bot never waits for an exchange to open.
SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"]
BENCHMARK = "BTC"
INTERVAL_MIN = 15
HISTORY_DAYS = 1095               # 3 years: bull, bear and sideways markets
WARMUP_CANDLES = 300              # candles needed before features are valid

# Paper account
START_CASH = 1000.0
FEE = 0.001                       # 0.1% per side (Binance spot taker fee)
SLIPPAGE = 0.0005                 # 0.05% worse fill per side
MIN_ORDER = 10.0

# Risk management
BASE_POSITION = 0.25              # fraction of equity per full-size position
MAX_EXPOSURE = 1.0                # never more than 100% of equity invested (no leverage)
DAILY_LOSS_PAUSE = 0.05           # pause new entries for 12h after -5% in 24h
PAUSE_HOURS = 12
DRAWDOWN_HALF_RISK = 0.15         # halve position size while >15% below the equity peak
SYMBOL_BENCH_TRUST = 0.6          # bench a symbol for 24h when its trust drops below this

# Trading styles the model can choose between (triple-barrier exits).
# tp/sl are multiples of the expected move over the horizon (volatility * sqrt(horizon)).
# (Styles shorter than ~16h were tested and dropped: fees ate every edge.)
VARIANTS = {
    "trend": {"horizon": 64, "tp": 1.8, "sl": 0.9},      # up to 16h
    "position": {"horizon": 192, "tp": 2.5, "sl": 1.2},  # up to 2 days
    "long": {"horizon": 384, "tp": 3.0, "sl": 1.5},      # up to 4 days
}
MIN_TP = 0.008                    # never target less than +0.8% (fees would eat it)
MIN_SL = 0.005

# Learning
RETRAIN_HOURS = 12
VALIDATION_FRACTION = 0.5        # last 50% of history (~1.5 years) is tested out-of-sample...
VALIDATION_FOLDS = 6              # ...in 6 walk-forward steps (train on the past, test the next block)
RECENCY_HALF_LIFE_DAYS = 60       # newer market data counts more
MIN_VALIDATION_TRADES = 40
TRAIN_STRIDE = 2                  # train on every 2nd candle (neighbours are near-duplicates): 2x faster
THRESHOLDS = [round(0.50 + 0.02 * i, 2) for i in range(16)]  # 0.50 .. 0.80

# Brain (LLM strategist)
BRAIN_DIRECTIVE_TTL_HOURS = 12    # stale advice expires, so a dead brain cannot freeze settings
LOCAL_BRAIN_MODEL = "mimo2.6:9b"          # fast, fits next to your other models in VRAM
CLOUD_BRAIN_MODEL = "qwen3:4b"
OLLAMA_URL = "http://127.0.0.1:11434"
CLM_ENABLED = True                # PC only: ask the local CLM for a second opinion every 4 hours
