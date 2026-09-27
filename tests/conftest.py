import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Keep tests away from the real portfolio files.
_tmp = Path(tempfile.mkdtemp(prefix="botdata-"))
os.environ["BOT_DATA_DIR"] = str(_tmp / "data")
os.environ["BOT_CACHE_DIR"] = str(_tmp / "cache")
(_tmp / "data").mkdir()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STEP = 15 * 60_000


def make_candles(n=4000, seed=0, start=1_700_000_000_000 // STEP * STEP, drift=0.0, vol=0.003):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, vol / 4, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({"ts": start + STEP * np.arange(n), "open": open_, "high": hi, "low": lo,
                         "close": close, "volume": rng.uniform(50, 150, n)})


@pytest.fixture
def candles():
    return make_candles()
