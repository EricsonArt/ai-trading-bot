"""The slow "brain": a local LLM (via Ollama) that reviews results and steers risk.

The fast model makes entry decisions; the brain cannot place trades. It reads a compact
report (portfolio, recent trades, model quality, market snapshot, its own earlier lessons
and how its last advice worked out) and returns bounded directives:
  risk_level          0.2 .. 1.0   scales every position size
  min_confidence_adj  -0.02 .. +0.10 raises/lowers the model's entry threshold
  avoid / reduce      symbols to skip or trade at half size
Directives expire after BRAIN_DIRECTIVE_TTL_HOURS, so a stale brain cannot freeze settings.
"""

import json
import time

import numpy as np
import requests

from . import config, data

BRAIN = config.DATA_DIR / "brain.json"
JOURNAL = config.DATA_DIR / "brain_journal.jsonl"
DEFAULTS = {"risk_level": 0.8, "min_confidence_adj": 0.0, "symbol_bias": {}, "regime": "unknown"}

SCHEMA = {
    "type": "object",
    "properties": {
        "regime": {"type": "string", "enum": ["bull", "bear", "sideways", "volatile"]},
        "risk_level": {"type": "number"},
        "min_confidence_adj": {"type": "number"},
        "avoid": {"type": "array", "items": {"type": "string", "enum": config.SYMBOLS}},
        "reduce": {"type": "array", "items": {"type": "string", "enum": config.SYMBOLS}},
        "lessons": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["regime", "risk_level", "min_confidence_adj", "avoid", "reduce", "lessons", "reasoning"],
}

SYSTEM = """You are the risk manager ("brain") of a crypto paper-trading bot that runs 24/7.
A fast machine-learning model makes all entry decisions; you cannot place trades yourself.
Your job: read the report, judge the market regime and the bot's recent performance, and set
risk directives for the next few hours. You also keep a short list of lessons learned.
Rules:
- Be skeptical and data-driven. Only use numbers that appear in the report.
- Reduce risk after losing streaks, in crashes, or when the model has no proven edge.
- Do not avoid a symbol without a concrete reason from the data.
- Lessons must be specific and actionable (max 3, each under 25 words).
- Reasoning: 2-4 plain sentences a beginner can understand.
Reply with JSON only."""


def _clip(x, lo, hi, default):
    try:
        return float(min(hi, max(lo, float(x))))
    except (TypeError, ValueError):
        return default


def sanitize(raw):
    """Clamp whatever the LLM produced into safe, bounded directives."""
    bias = {}
    for s in raw.get("reduce", []) or []:
        if s in config.SYMBOLS:
            bias[s] = "reduce"
    for s in (raw.get("avoid", []) or [])[:3]:  # never let the brain switch off the whole market
        if s in config.SYMBOLS:
            bias[s] = "avoid"
    regime = raw.get("regime") if raw.get("regime") in ("bull", "bear", "sideways", "volatile") else "unknown"
    return {
        "risk_level": round(_clip(raw.get("risk_level"), 0.2, 1.0, DEFAULTS["risk_level"]), 3),
        "min_confidence_adj": round(_clip(raw.get("min_confidence_adj"), -0.02, 0.10, 0.0), 3),
        "symbol_bias": bias,
        "regime": regime,
    }


def active_directives(now_ms=None):
    now_ms = now_ms or int(time.time() * 1000)
    b = _load(BRAIN, {})
    if not b.get("directives") or now_ms - b.get("created_at", 0) > config.BRAIN_DIRECTIVE_TTL_HOURS * 3_600_000:
        return dict(DEFAULTS)
    return {**DEFAULTS, **b["directives"]}


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _fmt_ts(ms):
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ms / 1000))


def market_snapshot():
    lines = []
    for s in config.SYMBOLS:
        df = data.load(s)
        if len(df) < 3000 or df.ts.iloc[-1] < data.now_ms() - 86_400_000:
            end = data.now_ms() // data.STEP_MS * data.STEP_MS
            df = data.fetch(s, end - 31 * 86_400_000, end)
        c = df.close.to_numpy(float)
        if len(c) < 2900:
            continue
        ch = lambda k: 100 * (c[-1] / c[-1 - k] - 1)
        vol = 100 * np.std(np.diff(np.log(c[-96:]))) * np.sqrt(96)
        lines.append(f"{s}: price {c[-1]:.6g}, 24h {ch(96):+.1f}%, 7d {ch(672):+.1f}%, 30d {ch(2880):+.1f}%, "
                     f"daily volatility {vol:.1f}%")
    return lines


def _trade_lines(trades, label):
    lines = [f"\n# {label}: last {min(len(trades), 10)} closed trades (of {len(trades)} total)"]
    for t in trades[-10:]:
        lines.append(f"{_fmt_ts(t['exit_ts'])} {t['symbol']}: {100 * t['net']:+.2f}% ({t['reason']}, "
                     f"model confidence {t.get('prob', 0):.2f})")
    week = [t for t in trades if t["exit_ts"] > data.now_ms() - 7 * 86_400_000]
    if week:
        wins = sum(t["net"] > 0 for t in week)
        lines.append(f"Last 7 days: {len(week)} trades, {wins} wins, total P/L ${sum(t['pnl'] for t in week):+.2f}")
    return lines


def build_report():
    state = _load(config.DATA_DIR / "state.json", {})
    prev = _load(BRAIN, {})
    main = state.get("main", {})
    acct = main.get("acct", {})
    last_eq = practice_eq = config.START_CASH
    eq_file = config.DATA_DIR / "equity.csv"
    if eq_file.exists():
        tail = eq_file.read_text(encoding="utf-8").strip().splitlines()[-1].split(",")
        if tail[0] != "ts":
            last_eq, practice_eq = float(tail[1]), float(tail[4])
    lines = ["# Main account (paper money; trades only while the model has a proven edge)",
             f"Start ${config.START_CASH:.0f} -> now ${last_eq:.2f} ({100 * (last_eq / config.START_CASH - 1):+.2f}%), "
             f"peak ${main.get('peak', last_eq):.2f}, cash ${acct.get('cash', last_eq):.2f}"]
    for s, p in acct.get("positions", {}).items():
        lines.append(f"Open: {s} bought {_fmt_ts(p['entry_ts'])} at {p['entry_px']:.6g}, "
                     f"stop {p['sl_px']:.6g}, target {p['tp_px']:.6g}, held {p['held']} candles")
    lines += _trade_lines(_read_jsonl(config.DATA_DIR / "trades.jsonl"), "Main account")
    lines += ["\n# Practice account (takes every model signal, ignores your advice; measures the raw model)",
              f"Start ${config.START_CASH:.0f} -> now ${practice_eq:.2f} "
              f"({100 * (practice_eq / config.START_CASH - 1):+.2f}%)"]
    lines += _trade_lines(_read_jsonl(config.DATA_DIR / "practice_trades.jsonl"), "Practice account")
    m = state.get("model", {})
    if m:
        mm = m["metrics"][m["variant"]]
        lines += ["\n# Fast model (tested on ~1.5 years of unseen data)",
                  f"Style '{m['variant']}', entry threshold {m['threshold']}, proven edge: {m['edge']}",
                  f"Out-of-sample: {mm['trades']} trades, win rate {100 * mm['win_rate']:.0f}%, "
                  f"avg {100 * mm['avg']:+.2f}%/trade vs always-buy {100 * mm['baseline_avg']:+.2f}%/trade",
                  "(Profit targets are about 2x the stop distance, so win rates below 50% can still be "
                  "profitable. Judge by average return per trade versus always-buy.)"]
    sig = main.get("signals", {})
    if sig:
        lines.append("\n# Model's current view (probability the trade ends in profit)")
        lines += [f"{s}: {v['p']:.2f} (threshold {v['threshold']}) -> {v['action']}" for s, v in sig.items()]
    lines.append("\n# Market")
    lines += market_snapshot()
    if prev.get("directives"):
        d = prev["directives"]
        lines += ["\n# Your previous advice", f"Given {_fmt_ts(prev['created_at'])}: risk_level {d['risk_level']}, "
                  f"threshold adjustment {d['min_confidence_adj']:+.2f}, symbol bias {d['symbol_bias'] or 'none'}"]
        if prev.get("equity_at"):
            lines.append(f"Since then the portfolio moved {100 * (last_eq / prev['equity_at'] - 1):+.2f}%.")
        if prev.get("lessons"):
            lines.append("Your lessons so far: " + " | ".join(prev["lessons"]))
    strat = state.get("strategy") or {}
    book = _load(config.DATA_DIR / "rules.json", {})
    act = [r for r in book.get("rules", []) if r["status"] == "active"]
    lines.append("\n# What the bot has learned (rules tested on unseen data)")
    lines.append(f"{len(book.get('rules', []))} ideas tested so far, {len(act)} active: "
                 + ("; ".join(f"{r['name']} (+{100 * r['test']['gain']:.2f}%/trade)" for r in act) or "none yet"))
    if strat:
        lines.append(f"Model + rules on unseen data: {strat['trades']} trades, avg {100 * strat['avg']:+.2f}%/trade, "
                     f"edge proven: {strat['edge']}")
    mistakes = _load(config.DATA_DIR / "research" / "mistakes.json", {})
    if mistakes.get("patterns"):
        lines.append("What losing signals had in common: " + "; ".join(
            f"{m['meaning']} (losers {m['losers_avg']:.3g} vs winners {m['winners_avg']:.3g})" for m in mistakes["patterns"][:3]))
    digest = _load(config.DATA_DIR / "research" / "digest.json", {})
    if digest.get("digest"):
        lines.append(f"\n# Today's research digest ({_fmt_ts(digest['created_at'])})\n{digest['digest']}")
        if digest.get("fng"):
            lines.append(f"Fear & Greed index: {digest['fng'][0]['value']} ({digest['fng'][0]['label']})")
    clm = _load(config.DATA_DIR / "clm.json", {})
    if clm.get("latest") and data.now_ms() - clm.get("updated_at", 0) < 6 * 3_600_000:
        lines.append("\n# Second opinion from the CLM model (fast text judge, still being evaluated)")
        lines += [f"{s}: {100 * p:.0f}% favourable" for s, p in clm["latest"].items()]
        test = _load(config.DATA_DIR / "clm_test.json", {})
        if test:
            lines.append(f"CLM historical test ({test['calls']} calls over {test['days']} days): accuracy "
                         f"{100 * test['accuracy']:.0f}% vs {100 * test['always_no_accuracy']:.0f}% for always saying no; "
                         f"ranking skill (AUC) {test['auc']:.2f} where 0.50 = no skill")
        if clm.get("track_record"):
            tr = clm["track_record"]
            lines.append(f"CLM track record: {tr['n']} checked calls, accuracy {100 * tr['accuracy']:.0f}% "
                         f"(the market went up {100 * tr['base_rate']:.0f}% of the time)")
    lines.append(
        "\n# Your task\nReturn JSON with: regime; risk_level (0.2-1.0, default 0.8); "
        "min_confidence_adj (-0.02 to 0.10, positive = stricter entries); avoid (symbols to skip, max 3); "
        "reduce (symbols at half size); lessons (max 3); reasoning.")
    return "\n".join(lines), last_eq


def ask(model_name, report, system=SYSTEM, schema=SCHEMA, timeout=900):
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
        "model": model_name, "stream": False, "think": False, "format": schema, "keep_alive": "2m",
        "options": {"temperature": 0.3, "num_ctx": 12288},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": report}],
    }, timeout=timeout)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    return json.loads(content[content.find("{"): content.rfind("}") + 1])


def run(model_name, source):
    report, equity = build_report()
    started = time.time()
    raw = ask(model_name, report)
    directives = sanitize(raw)
    prev = _load(BRAIN, {})
    lessons = [str(x).strip()[:200] for x in raw.get("lessons", []) if str(x).strip()][:3]
    memory = (lessons + [x for x in prev.get("lessons", []) if x not in lessons])[:12]
    now = int(time.time() * 1000)
    entry = {
        "created_at": now, "source": source, "model": model_name, "seconds": round(time.time() - started, 1),
        "directives": directives, "reasoning": str(raw.get("reasoning", ""))[:1200],
        "new_lessons": lessons, "lessons": memory, "equity_at": equity,
    }
    BRAIN.parent.mkdir(parents=True, exist_ok=True)
    BRAIN.write_text(json.dumps(entry, indent=1), encoding="utf-8")
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: entry[k] for k in ("created_at", "source", "model", "directives",
                                                   "reasoning", "new_lessons", "equity_at")}) + "\n")
    print(f"[brain] {source}/{model_name} in {entry['seconds']}s: {directives}")
    return entry


def fresh(hours):
    """True if the brain already ran within `hours` (lets the cloud skip when the PC just did it)."""
    b = _load(BRAIN, {})
    return int(time.time() * 1000) - b.get("created_at", 0) < hours * 3_600_000
