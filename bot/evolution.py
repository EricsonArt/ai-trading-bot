"""Strategy evolution: the bot's "dream map".

A strategy recipe ("genome") = which trading style's model to follow, how confident it must
be before buying, take-profit / stop-loss / holding time, position size, and entry rules.

Every retrain the bot "dreams": it replays ~1.5 years of recorded history (the model's
out-of-sample predictions + real prices) under hundreds of recipes without trading them.
New recipes come from mutating and crossing the best ones found so far, from rules mined
out of its own losing trades, and from rules suggested by its daily internet research.
Every recipe ever tried is written to a map (data/evolution/archive.jsonl) with its parent,
so the search keeps building on what worked and never re-learns what failed.

To keep it from fooling itself (try enough recipes and some look great by luck), the
evidence is split in time: recipes compete on the older 60%, the finalists on the next
20%, and a new recipe replaces the champion only if it also wins on the newest 20%, which
the search never looked at.
"""

import hashlib
import json
import time

import numpy as np

from . import config, features, rules
from .model import simulate, split_edges, trade_stats

DIR = config.DATA_DIR / "evolution"
ARCHIVE = DIR / "archive.jsonl"
CHAMPION = DIR / "champion.json"
MAP = DIR / "map.json"
IDEAS = DIR / "ideas.json"
STEP_MS = config.INTERVAL_MIN * 60_000
E = config.EVOLUTION
RANGES = {"threshold": (E["min_threshold"], 0.92), "tp": (0.6, 4.0), "sl": (0.4, 3.0),
          "size": (0.10, 0.25)}  # size capped: bigger bets deepened drawdowns in testing
HOLDS = [0.5, 0.75, 1.0, 1.5, 2.0]
MAX_RULES = 4
GENES = ("style", "threshold", "tp", "sl", "hold", "size")


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


# ---------- recipes ----------

def normalize(g):
    """Clip genes to their ranges, snap them to a grid, dedupe rules, stamp an id."""
    out = {k: g[k] for k in GENES}
    out["threshold"] = round(float(np.clip(g["threshold"], *RANGES["threshold"])), 2)
    out["tp"] = round(float(np.clip(g["tp"], *RANGES["tp"])) * 5) / 5
    out["sl"] = round(float(np.clip(g["sl"], *RANGES["sl"])) * 5) / 5
    out["hold"] = min(HOLDS, key=lambda x: abs(x - g["hold"]))
    out["size"] = round(float(np.clip(g["size"], *RANGES["size"])), 2)
    seen, conds = set(), []
    for c in g.get("rules", []):
        if (c["feature"], c["op"]) not in seen and len(conds) < MAX_RULES:
            seen.add((c["feature"], c["op"]))
            conds.append({"feature": c["feature"], "op": c["op"], "value": round(float(c["value"]), 5)})
    out["rules"] = conds
    key = json.dumps({**{k: out[k] for k in GENES}, "rules": sorted(json.dumps(c, sort_keys=True) for c in conds)})
    out["id"] = hashlib.sha1(key.encode()).hexdigest()[:10]
    out["parent"], out["origin"] = g.get("parent"), g.get("origin", "")
    return out


def default_recipe(bundle, style):
    v = config.VARIANTS[style]
    return normalize({"style": style, "threshold": max(bundle["metrics"][style]["threshold"], E["min_threshold"]),
                      "tp": v["tp"], "sl": v["sl"], "hold": 1.0, "size": config.BASE_POSITION, "rules": [],
                      "origin": "model default"})


def horizon(g):
    return max(8, int(round(config.VARIANTS[g["style"]]["horizon"] * g["hold"])))


def describe(g):
    parts = [f"{g['style']} style", f"buy when the model is at least {round(100 * g['threshold'])}% sure",
             f"target {g['tp']:g}x / stop {g['sl']:g}x the expected move",
             f"hold up to {horizon(g) * config.INTERVAL_MIN / 60:.0f} h", f"{round(100 * g['size'])}% of equity per trade"]
    if g["rules"]:
        parts.append("only if " + rules.describe(g["rules"]))
    return ", ".join(parts)


def as_rules(g):
    """The recipe's conditions in the form the live trader checks (one named rule each)."""
    return [{"name": rules.describe([c]), "conditions": [c]} for c in g["rules"]]


def mutate(g, rng, pool, ranges):
    c = {**g, "rules": [dict(x) for x in g["rules"]]}
    op = rng.choice(["threshold", "threshold", "tp", "sl", "hold", "size", "add_rule", "add_rule",
                     "drop_rule", "tweak_rule", "style"])
    if op == "threshold":
        c["threshold"] += rng.choice([-0.04, -0.02, -0.01, 0.01, 0.02, 0.04])
    elif op in ("tp", "sl"):
        c[op] *= rng.choice([0.7, 0.85, 1.15, 1.3])
    elif op == "hold":
        c["hold"] = float(rng.choice(HOLDS))
    elif op == "size":
        c["size"] += rng.choice([-0.05, 0.05])
    elif op == "add_rule":
        if pool and rng.random() < 0.5:
            c["rules"] += [dict(x) for x in pool[rng.integers(len(pool))]]
        else:
            f = str(rng.choice(list(ranges)))
            q = ranges[f]
            c["rules"].append({"feature": f, "op": str(rng.choice(["<", ">"])), "value": rng.uniform(q[0], q[2])})
    elif op == "drop_rule" and c["rules"]:
        c["rules"].pop(int(rng.integers(len(c["rules"]))))
    elif op == "tweak_rule" and c["rules"]:
        r = c["rules"][int(rng.integers(len(c["rules"])))]
        q = ranges.get(r["feature"], [0, 0, 0])
        r["value"] += rng.normal(0, 0.15 * ((q[2] - q[0]) or abs(r["value"]) or 1e-3))
    elif op == "style":
        c["style"] = str(rng.choice([s for s in config.VARIANTS if s != g["style"]]))
    c["parent"], c["origin"] = g["id"], f"mutation: {op}"
    return normalize(c)


def crossover(a, b, rng):
    c = {k: (a if rng.random() < 0.5 else b)[k] for k in ("tp", "sl", "hold", "size")}
    c["style"], c["threshold"] = a["style"], a["threshold"]  # confidence levels are model-specific
    c["rules"] = [dict(x) for x in a["rules"] + b["rules"] if rng.random() < 0.5]
    c["parent"], c["origin"] = f"{a['id']}+{b['id']}", "crossover"
    return normalize(c)


# ---------- dreaming ----------

def outcome_stats(trades, size):
    """Per-trade stats plus a portfolio growth score (log growth minus one standard error)."""
    st = trade_stats(trades)
    g = np.log1p(size * trades)
    n = len(g)
    score = float(n * g.mean() - np.sqrt(n) * (g.std(ddof=1) if n > 1 else abs(g[0]))) if n else -1.0
    return {"trades": n, "avg": round(st["avg"], 5), "win_rate": round(st["win_rate"], 4),
            "growth": round(float(g.sum()), 5), "score": round(score, 5)}


class Dreamer:
    """Replays recorded history under any recipe, without trading."""

    def __init__(self, bundle, candles):
        self.oos = bundle["oos"]
        self.px = {s: (int(df.ts.iloc[0]), *(df[k].to_numpy(float) for k in ("open", "high", "low", "close")))
                   for s, df in candles.items() if len(df)}
        self.edges = split_edges(np.concatenate([d.ts.to_numpy() for d in self.oos.values()]))
        self._exits = {}
        self.evaluated = 0

    def exits(self, style, tp, sl, hold):
        """Net result and duration of every recorded signal under these exit settings (cached)."""
        key = (style, tp, sl, hold)
        if key not in self._exits:
            df = self.oos[style]
            H = max(8, int(round(config.VARIANTS[style]["horizon"] * hold)))
            net, off = np.full(len(df), np.nan), np.full(len(df), np.nan)
            ts, vol = df.ts.to_numpy(), df.vol_96.to_numpy(float)
            for sym, rows in df.groupby("symbol").indices.items():
                if sym not in self.px:
                    continue
                ts0, o, h, l, c = self.px[sym]
                idx = (ts[rows] - ts0) // STEP_MS
                ok = (idx >= 0) & (idx < len(c))
                rows, idx = rows[ok], idx[ok].astype(np.int64)
                tpf, slf = features.barriers(vol[rows], H, tp, sl)
                n_, o_, resolved = features.triple_barrier(o, h, l, c, idx, tpf, slf, H)
                net[rows] = np.where(resolved, n_, np.nan)
                off[rows] = o_
            self._exits[key] = (net, off)
        return self._exits[key]

    def evaluate(self, g):
        self.evaluated += 1
        df = self.oos[g["style"]]
        net, off = self.exits(g["style"], g["tp"], g["sl"], g["hold"])
        keep = (df.p.to_numpy() >= g["threshold"]) & ~np.isnan(net) & rules.mask([{"conditions": g["rules"]}], df)
        ts, sym = df.ts.to_numpy(), df.symbol.to_numpy()
        a, b = self.edges
        out = {}
        for name, period in (("search", ts < a), ("select", (ts >= a) & (ts < b)), ("holdout", ts >= b)):
            k = keep & period
            trades = simulate(sym[k], ts[k], net[k], off[k], np.ones(int(k.sum())), 0.5)
            out[name] = outcome_stats(trades, g["size"])
        return out


def _fitness(item, split="search"):
    st = item[1][split]
    return st["score"] if st["trades"] >= E["min_trades"][split] else -1e6 + st["trades"]


def _archive_elites(n):
    if not ARCHIVE.exists():
        return []
    best = {}
    for line in ARCHIVE.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        best[r["id"]] = r  # the latest record of each recipe wins
    ranked = sorted(best.values(), key=lambda r: r["stats"]["select"]["score"], reverse=True)
    return [r["genome"] for r in ranked[:n]]


def run(bundle, candles, ideas=(), budget_s=None, seed=None, persist=True, champion=None):
    """One dreaming session: replay, evolve, maybe deploy a new champion. Returns the champion.

    persist=False keeps everything in memory (used by the backtest; pass the previous champion)."""
    t0 = time.time()
    budget = E["budget_seconds"] if budget_s is None else budget_s
    dreamer = Dreamer(bundle, candles)
    world = _load(MAP, {"generations": [], "deployments": []}) if persist else {"generations": [], "deployments": []}
    gen = (world["generations"][-1]["gen"] + 1) if world["generations"] else 1
    rng = np.random.default_rng(seed if seed is not None else gen * 7919 + bundle["version"])
    ranges = {f: q for f, q in bundle["feature_ranges"].items() if f in rules.FEATURE_INFO}
    old = _load(CHAMPION, None) if persist else champion

    # the bot's own mistakes (under the current champion's exits) become ideas too
    base_recipe = normalize(old["genome"]) if old and old["genome"]["style"] in bundle["oos"] \
        else default_recipe(bundle, bundle["variant"])
    net, _ = dreamer.exits(base_recipe["style"], base_recipe["tp"], base_recipe["sl"], base_recipe["hold"])
    ideas = list(ideas) + rules.mine_mistakes(dreamer.oos[base_recipe["style"]], net, base_recipe["threshold"],
                                             dreamer.edges[0], bundle["version"], write=persist)
    pool = [i["conditions"] for i in ideas]

    scored = {}

    def add(g):
        if g["id"] not in scored and g["style"] in dreamer.oos:
            scored[g["id"]] = (g, dreamer.evaluate(g))
        return g

    for style in dreamer.oos:
        add(default_recipe(bundle, style))
    champ_recipe = add(base_recipe)
    for g in (_archive_elites(40) if persist else []):
        add(normalize(g))
    idea_recipes = {}
    for idea in ideas:  # each idea is tried on top of the current champion
        g = add(normalize({**base_recipe, "rules": base_recipe["rules"] + idea["conditions"],
                           "parent": base_recipe["id"], "origin": f"idea: {idea['name']}"}))
        idea_recipes[idea["id"]] = (idea, g["id"])

    rounds = 0
    while time.time() - t0 < budget and rounds < 40:
        parents = [g for g, _ in sorted(scored.values(), key=_fitness, reverse=True)[:E["population"]]]
        for _ in range(E["offspring"]):
            if time.time() - t0 > budget:
                break
            if len(parents) > 1 and rng.random() < 0.3:
                i, j = rng.choice(len(parents), 2, replace=False)
                add(crossover(parents[i], parents[j], rng))
            else:
                add(mutate(parents[min(int(rng.exponential(5)), len(parents) - 1)], rng, pool, ranges))
        rounds += 1

    ranked = sorted(scored.values(), key=_fitness, reverse=True)
    finalists = [x for x in ranked[:12] if x[1]["select"]["trades"] >= E["min_trades"]["select"]] or ranked[:1]
    cand = max(finalists, key=lambda x: _fitness(x, "select"))
    champ = scored[champ_recipe["id"]]
    ch, hh = cand[1]["holdout"], champ[1]["holdout"]
    if old is None:
        deploy = True  # first run: the best recipe found becomes the champion
    elif cand[0]["id"] == champ[0]["id"]:
        deploy = False
    else:  # a challenger must beat the champion on data the search never looked at
        deploy = (ch["trades"] >= E["min_trades"]["holdout"] and ch["score"] > hh["score"]
                  and ch["avg"] >= hh["avg"] + 0.0005 and _fitness(cand, "select") > _fitness(champ, "select"))
    winner = cand if deploy else champ
    g, st = winner
    base = bundle["baselines"][g["style"]]
    edge = (st["select"]["score"] > 0 and st["holdout"]["score"] > 0 and st["holdout"]["trades"] >= E["min_trades"]["holdout"]
            and st["holdout"]["avg"] > max(0.0, base["holdout"]) + 0.001)
    now = int(time.time() * 1000)
    champion = {
        "genome": g, "stats": st, "edge": bool(edge), "baseline": base, "description": describe(g),
        "model_version": bundle["version"], "generation": gen,
        "since": now if deploy or old is None or old["genome"]["id"] != g["id"] else old.get("since", now),
    }
    if not persist:
        return champion
    _save(CHAMPION, champion)

    # write the map: the best recipes of this session (+ challenger and champion), with lineage
    keep = {x[0]["id"]: x for x in ranked[:15] + [cand, winner]}
    DIR.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open("a", encoding="utf-8") as f:
        for gid, (rg, rs) in keep.items():
            f.write(json.dumps({"id": gid, "gen": gen, "ts": now, "genome": rg, "stats": rs,
                                "deployed": bool(deploy and gid == g["id"])}) + "\n")
    tried = [st["select"]["avg"] for _, st in scored.values() if st["select"]["trades"] >= E["min_trades"]["select"]]
    sample = sorted(rng.choice(tried, min(80, len(tried)), replace=False).tolist()) if tried else []
    world["generations"].append({
        "gen": gen, "ts": now, "model_version": bundle["version"], "evaluated": len(scored), "rounds": rounds,
        "sample": [round(x, 5) for x in sample],  # what the whole search looked like (for the map)
        "seconds": round(time.time() - t0, 1), "best_search_avg": ranked[0][1]["search"]["avg"],
        "challenger_select_avg": cand[1]["select"]["avg"], "champion_holdout_avg": st["holdout"]["avg"],
        "champion_id": g["id"], "deployed": bool(deploy), "edge": bool(edge)})
    world["generations"] = world["generations"][-400:]
    if deploy:
        world["deployments"] = (world["deployments"] + [{"gen": gen, "ts": now, "id": g["id"], "description": describe(g),
                                                         "genome": g,
                                                         "holdout_avg": st["holdout"]["avg"]}])[-50:]
    world["total_evaluated"] = world.get("total_evaluated", 0) + len(scored)
    _save(MAP, world)

    # verdict for every idea: did adding it to the champion help on the select period?
    verdicts = _load(IDEAS, {})
    for iid, (idea, rid) in idea_recipes.items():
        if iid in verdicts and verdicts[iid].get("model_version") == bundle["version"]:
            continue
        gain = scored[rid][1]["select"]["avg"] - champ[1]["select"]["avg"]
        verdicts[iid] = {"name": idea["name"], "source": idea["source"], "rule": rules.describe(idea["conditions"]),
                         "why": idea.get("why", ""), "gain": round(gain, 5), "tested_at": now,
                         "model_version": bundle["version"],
                         "adopted": all(c in g["rules"] for c in idea["conditions"])}
    _save(IDEAS, dict(sorted(verdicts.items(), key=lambda kv: kv[1]["tested_at"])[-80:]))

    print(f"[evolution] gen {gen}: dreamed {len(scored)} recipes in {time.time() - t0:.0f}s; "
          f"champion {'NEW' if deploy else 'kept'}: {describe(g)} | holdout {100 * st['holdout']['avg']:+.2f}%/trade "
          f"over {st['holdout']['trades']} trades, edge={edge}")
    return champion


def load_champion():
    return _load(CHAMPION, None)
