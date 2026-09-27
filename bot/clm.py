"""Optional second opinion from the local CLM (Contrastive Language Model, CLM-v0.1-8B).

CLM is a fast "System One" text judge: it scores how true a statement is for a given
state. We describe each coin's recent market in plain words and ask whether buying now
and holding ~2 days will profit. CLM does not see the trading model's opinion, so its
calls are independent and can be scored honestly: every call is logged and later checked
against what the price actually did. The brain sees CLM's live track record and decides
how much to trust it. CLM only runs on the PC (it needs the 16 GB encoder weights).
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import requests

from . import config, data

HOME = Path(os.environ.get("CLM_HOME", r"C:\Users\cosmi\Documents\Codex\2026-09-26\pobierze-mi-na-m-j-komputer"))
WORK = HOME / "work"
WEIGHTS = HOME / "outputs" / "Modele lokalne" / "06_Ocena_i_wybor" / "CLM-v0.1-8B"
PYTHON = WORK / "clm-venv" / "Scripts" / "python.exe"
SERVE = WORK / "clm-venv" / "Scripts" / "clm-serve.exe"
API = "http://127.0.0.1:8700"
ENCODER = "http://127.0.0.1:8090"
LOG = config.DATA_DIR / "clm_log.jsonl"
SUMMARY = config.DATA_DIR / "clm.json"
HOLD = 192  # candles (2 days) — the horizon each call is judged on
QUESTION = "Buying {sym} now and holding it for about two days will end in a profit."


def installed():
    return PYTHON.exists() and SERVE.exists() and (WEIGHTS / "CLM_v0.1-8B.pt").exists()


def _up(url):
    try:
        return requests.get(url + "/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def _start(args, url, seconds, log_name):
    log = open(config.CACHE_DIR / log_name, "ab")
    proc = subprocess.Popen(args, cwd=HOME, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(seconds // 5):
        if _up(url):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"CLM process exited early; see .cache/{log_name}")
        time.sleep(5)
    _stop(proc)
    raise TimeoutError(f"CLM did not start within {seconds}s")


def _stop(proc):
    if proc and proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)


def describe(sym, df, btc):
    """Plain-language market state (CLM's heads are trained on prose, not numbers in JSON)."""
    c, v = df.close.to_numpy(float), df.volume.to_numpy(float)
    b = btc.close.to_numpy(float)
    ch = lambda x, k: f"{100 * (x[-1] / x[-1 - k] - 1):+.1f}%"
    delta = np.diff(c[-15:])
    up, down = delta.clip(min=0).mean(), (-delta).clip(min=0).mean()
    rsi = 100 * up / (up + down) if up + down else 50
    ma200h = c[-800:].mean()
    return {
        "asset": f"{sym} (cryptocurrency, priced in US dollars)",
        "time": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(df.ts.iloc[-1] / 1000)),
        "price": f"{c[-1]:.6g}",
        "price change": f"1 hour {ch(c, 4)}, 24 hours {ch(c, 96)}, 7 days {ch(c, 672)}, 30 days {ch(c, 2880)}",
        "daily volatility": f"{100 * np.std(np.diff(np.log(c[-96:]))) * np.sqrt(96):.1f}%",
        "momentum (RSI, 0-100)": f"{rsi:.0f}",
        "trend": f"price is {100 * (c[-1] / ma200h - 1):+.1f}% versus its 200-hour average",
        "trading volume": f"{v[-96:].sum() / max(v[-672:].sum() / 7, 1e-9):.2f}x the average day",
        "bitcoin": f"24 hours {ch(b, 96)}, 7 days {ch(b, 672)}",
    }


def ask(states):
    out = {}
    for sym, state in states.items():
        r = requests.post(API + "/v1/systemone", json={
            "state": state, "questions": {"profit": {"type": "noul", "instructions": QUESTION.format(sym=sym)}},
        }, timeout=300)
        r.raise_for_status()
        out[sym] = float(r.json()["answers"]["profit"]["noul"])
    return out


def score_history(candles):
    """Check old calls against what actually happened, return CLM's track record."""
    rows = [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()] if LOG.exists() else []
    hits, ups = [], []
    for r in rows:
        df = candles.get(r["symbol"])
        if df is None:
            continue
        later = df[df.ts == r["ts"] + HOLD * data.STEP_MS]
        if later.empty:
            continue
        went_up = float(later.close.iloc[0]) > r["price"] * (1 + 2 * (config.FEE + config.SLIPPAGE))
        hits.append((r["p"] > 0.5) == went_up)
        ups.append(went_up)
    if not hits:
        return None
    return {"n": len(hits), "accuracy": float(np.mean(hits)), "base_rate": float(np.mean(ups))}


def run_once():
    """Start CLM if needed, ask about every coin, log, stop what we started. Returns summary or None."""
    if not installed():
        print("[clm] not installed; skipping")
        return None
    candles = {s: data.update(s) for s in config.SYMBOLS}
    started = []
    try:
        if not _up(ENCODER):
            started.append(_start([str(PYTHON), str(WORK / "clm-encoder-server.py")], ENCODER, 900, "clm-encoder.log"))
        if not _up(API):
            started.append(_start([str(SERVE), "--host", "127.0.0.1", "--port", "8700", "--emb-url",
                                   ENCODER + "/v1/embeddings", "--ckpt", str(WEIGHTS / "CLM_v0.1-8B.pt"),
                                   "--no-download", "--device", "cpu", "--action-cache", "0", "--no-ui"],
                                  API, 180, "clm-app.log"))
        t0 = time.time()
        answers = ask({s: describe(s, df, candles[config.BENCHMARK]) for s, df in candles.items()})
        seconds = round(time.time() - t0, 1)
    finally:
        for proc in reversed(started):
            _stop(proc)
    with LOG.open("a", encoding="utf-8") as f:
        for s, p in answers.items():
            df = candles[s]
            f.write(json.dumps({"ts": int(df.ts.iloc[-1]), "symbol": s, "p": round(p, 4),
                                "price": float(df.close.iloc[-1])}) + "\n")
    summary = {"updated_at": int(time.time() * 1000), "latest": {s: round(p, 4) for s, p in answers.items()},
               "seconds": seconds, "question": QUESTION, "track_record": score_history(candles)}
    SUMMARY.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[clm] answered in {seconds}s: {summary['latest']}")
    return summary
