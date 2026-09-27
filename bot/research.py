"""Daily internet research: the brain reads fresh trading knowledge and proposes rules.

Sources (free, no keys): new trading research on arXiv, crypto news (CoinDesk,
Cointelegraph, Decrypt), Babypips trading lessons, and the Crypto Fear & Greed index.
The LLM also sees what the bot's own losing trades had in common (rules.mine_mistakes).
It writes a short digest and up to 3 testable rules. Nothing read online is applied
directly: proposals go to data/research/proposals.jsonl, and the cloud bot judges each one
on ~1.5 years of unseen data (rules.judge) before it can influence a single trade.
"""

import json
import re
import time
import xml.etree.ElementTree as ET

import requests

from . import brain, config, rules

DIR = config.DATA_DIR / "research"
DIGEST = DIR / "digest.json"
HISTORY = DIR / "digests.jsonl"
UA = {"User-Agent": "Mozilla/5.0 (paper-trading-bot research)"}
NEWS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}
LESSONS = {"Babypips": "https://www.babypips.com/feed.rss"}
ARXIV = ("https://export.arxiv.org/api/query?search_query=(cat:q-fin.TR+OR+cat:q-fin.ST+OR+cat:q-fin.PM)"
         "+AND+(abs:crypto+OR+abs:bitcoin+OR+abs:cryptocurrency+OR+abs:trading)"
         "&sortBy=submittedDate&sortOrder=descending&max_results=6")

SYSTEM = """You are the research analyst of a crypto paper-trading bot.
A machine-learning model generates buy signals; your job is to find FILTERS that would have
removed its bad signals. You read today's trading research, news and lessons, plus a summary of
what the bot's own losing trades had in common. Then propose rules the bot can test.
Rules:
- Each rule is 1-3 conditions on the listed features only, like {"feature": "rsi", "op": "<", "value": 0.7}.
- Keep thresholds inside the typical ranges given. A rule keeps a signal only if ALL conditions are true.
- Ground every rule in something you read or in the bot's mistakes; say which in "source" and "why".
- Do not repeat rules that are already active or were rejected.
- The digest: 3-5 plain sentences on what matters for crypto traders today.
Reply with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "digest": {"type": "string"},
        "mood": {"type": "string", "enum": ["fear", "neutral", "greed"]},
        "ideas": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}, "why": {"type": "string"}, "source": {"type": "string"},
                "conditions": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"feature": {"type": "string", "enum": sorted(rules.FEATURE_INFO)},
                                   "op": {"type": "string", "enum": ["<", ">"]},
                                   "value": {"type": "number"}},
                    "required": ["feature", "op", "value"]}},
            },
            "required": ["name", "why", "source", "conditions"]}},
    },
    "required": ["digest", "mood", "ideas"],
}


def _text(el, tag, ns=None):
    x = el.find(tag, ns) if ns else el.find(tag)
    return re.sub(r"<[^>]+>|\s+", " ", x.text or "").strip() if x is not None and x.text else ""


def _get(url):
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    return r.content


def _rss(name, url, limit, summary_chars):
    root = ET.fromstring(_get(url))
    items = []
    for it in root.iter("item"):
        items.append({"kind": name, "title": _text(it, "title"), "url": _text(it, "link"),
                      "summary": _text(it, "description")[:summary_chars]})
        if len(items) >= limit:
            break
    return items


def _arxiv():
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(_get(ARXIV))
    return [{"kind": "arXiv research", "title": _text(e, "a:title", ns), "url": _text(e, "a:id", ns),
             "summary": _text(e, "a:summary", ns)[:700]} for e in root.findall("a:entry", ns)]


def _fear_greed():
    d = json.loads(_get("https://api.alternative.me/fng/?limit=7"))["data"]
    return [{"value": int(x["value"]), "label": x["value_classification"], "ts": int(x["timestamp"]) * 1000} for x in d]


def gather():
    items, used, failed = [], [], []
    jobs = [("arXiv", _arxiv)] + [(n, lambda n=n, u=u: _rss(n, u, 6, 200)) for n, u in NEWS.items()] \
        + [(n, lambda n=n, u=u: _rss(n, u, 5, 350)) for n, u in LESSONS.items()]
    for name, fn in jobs:
        try:
            got = fn()
            items += got
            used.append(f"{name} ({len(got)})")
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}")
    try:
        fng = _fear_greed()
        used.append("Fear & Greed index")
    except Exception as exc:
        fng = []
        failed.append(f"Fear & Greed: {type(exc).__name__}")
    return {"items": items, "fng": fng, "used": used, "failed": failed}


def build_prompt(material):
    state = brain._load(config.DATA_DIR / "state.json", {})
    ranges = state.get("feature_ranges") or {}
    mistakes = brain._load(rules.MISTAKES, {})
    book = rules.load()
    lines = ["# What you read today"]
    for it in material["items"]:
        lines.append(f"- [{it['kind']}] {it['title']}" + (f": {it['summary']}" if it["summary"] else ""))
    if material["fng"]:
        lines.append("Crypto Fear & Greed index, last 7 days (newest first): "
                     + ", ".join(f"{x['value']} {x['label']}" for x in material["fng"]))
    if mistakes.get("patterns"):
        lines.append(f"\n# The bot's own mistakes ({mistakes['losers']} losing of {mistakes['signals']} tested signals)")
        lines += [f"- {p['meaning']} ({p['feature']}): losing trades avg {p['losers_avg']:.4g}, "
                  f"winning trades avg {p['winners_avg']:.4g}" for p in mistakes["patterns"]]
    lines.append("\n# Features you may use: name: meaning (typical range: 10% / median / 90%)")
    for f, meaning in rules.FEATURE_INFO.items():
        q = ranges.get(f)
        lines.append(f"- {f}: {meaning}" + (f" ({q[0]:.4g} / {q[1]:.4g} / {q[2]:.4g})" if q else ""))
    act = rules.active(book)
    rej = [r for r in book["rules"] if r["status"] != "active"][-10:]
    if act:
        lines.append("\n# Active rules (already in use)")
        lines += [f"- {r['name']}: {rules.describe(r['conditions'])}" for r in act]
    if rej:
        lines.append("\n# Recently rejected or retired rules (they did not help on unseen data)")
        lines += [f"- {rules.describe(r['conditions'])}" for r in rej]
    lines.append("\n# Task\nReturn JSON: digest, mood (fear/neutral/greed), ideas (0-3 rules).")
    return "\n".join(lines)


def run(model_name, source):
    material = gather()
    prompt = build_prompt(material)
    started = time.time()
    raw = brain.ask(model_name, prompt, system=SYSTEM, schema=SCHEMA)
    now = int(time.time() * 1000)
    ideas = []
    for idea in (raw.get("ideas") or [])[:3]:
        rule = rules.clean(idea, "internet research: " + str(idea.get("source", "")))
        if rule:
            ideas.append({**rule, "created_at": now})
    DIR.mkdir(parents=True, exist_ok=True)
    with rules.PROPOSALS.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in ideas)
    digest = {
        "created_at": now, "source": source, "model": model_name, "seconds": round(time.time() - started, 1),
        "digest": str(raw.get("digest", ""))[:1500], "mood": raw.get("mood"),
        "fng": material["fng"][:1], "used": material["used"], "failed": material["failed"],
        "items": [{k: it[k] for k in ("kind", "title", "url")} for it in material["items"]][:30],
        "ideas": [{"name": r["name"], "rule": rules.describe(r["conditions"]), "why": r["why"]} for r in ideas],
    }
    DIGEST.write_text(json.dumps(digest, indent=1), encoding="utf-8")
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: digest[k] for k in ("created_at", "source", "model", "digest", "mood", "ideas")}) + "\n")
    print(f"[research] {source}/{model_name}: read {len(material['items'])} items from {', '.join(material['used'])}; "
          f"proposed {len(ideas)} rule(s)")
    return digest


def fresh(hours):
    d = brain._load(DIGEST, {})
    return int(time.time() * 1000) - d.get("created_at", 0) < hours * 3_600_000
