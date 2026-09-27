"""Pre-registered analysis for the outside judge (soc 2026-09-27 10:35 + 11:10): base vs one adapter on a task set, paired
by task. Inputs are lab.run per-episode jsonl files (fields task, seed, pass, hit_cap, via, turns).

  uv run python -m lab.judge_stats --tasks lab/spider2_sqlite.json \\
      --base runs/spider2-eval/spider2-base.jsonl --treat runs/spider2-eval/spider2-step100.jsonl runs/spider2-eval/spider2-step100-s23.jsonl \\
      [--seen-schemas sqlite-sakila,Pagila,IPL,AdventureWorks,EU_soccer,f1] [--boot 20000]

Reports: per-arm pass rates by seed; Δ = mean over tasks of (p̂_treat − p̂_base); 95 % CI from a bootstrap over databases
(cluster resampling, percentile) — PRIMARY; exact McNemar on per-task seed-count wins/losses and a database sign-flip test
— SECONDARY; pre-declared strata (seen vs unseen schemas; base-reachable vs base-never tasks); transition decomposition
(no-submit→pass vs wrong→pass); no-submit and turn-cap rates. Decision rule: "beats base" iff the cluster-bootstrap lower
bound > 0; "meaningful" iff Δ ≥ +5 points. Replaying climb 1 (base ×3 vs step100 ×3) must give Δ ≈ +1.73, CI ≈ [−1.6, +5.0].
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def outcome(e):
    if e.get("pass"):
        return "pass"
    if e.get("hit_cap") or e.get("via") in (None, "error", "malformed"):
        return "nosub"
    return "wrong"


def per_task(rows):
    d = collections.defaultdict(list)
    for e in rows:
        d[e["task"]].append(e)
    return d


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b (treat better) and c (base better)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--treat", nargs="+", required=True)
    ap.add_argument("--seen-schemas", default="sqlite-sakila,Pagila,IPL,AdventureWorks,EU_soccer,f1",
                    help="Spider schemas that overlap pool/steering DBs (Opus review C2); pre-declared stratum")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=5.0, help="'meaningful' threshold in points")
    a = ap.parse_args()
    tasks = {t["id"]: t for t in json.load(open(a.tasks))}
    base, treat = per_task(load(a.base)), per_task(load(a.treat))
    ids = sorted(set(tasks) & set(base) & set(treat))
    print(f"tasks {len(ids)} · base episodes {sum(len(base[t]) for t in ids)} · treat episodes {sum(len(treat[t]) for t in ids)}")
    for name, arm in (("base", base), ("treat", treat)):
        by_seed = collections.defaultdict(lambda: [0, 0])
        for t in ids:
            for e in arm[t]:
                by_seed[e["seed"]][0] += bool(e.get("pass")); by_seed[e["seed"]][1] += 1
        seeds = {s: f"{p}/{n}" for s, (p, n) in sorted(by_seed.items())}
        n = sum(len(arm[t]) for t in ids); p = sum(1 for t in ids for e in arm[t] if e.get("pass"))
        oc = collections.Counter(outcome(e) for t in ids for e in arm[t])
        print(f"{name:5s} acc {p/n:.4f} · by seed {seeds} · no-submit {oc['nosub']/n:.3f} · wrong {oc['wrong']/n:.3f} · "
              f"turn-cap {sum(1 for t in ids for e in arm[t] if e.get('hit_cap'))/n:.3f} · turns {sum(e['turns'] for t in ids for e in arm[t])/n:.1f}")
    pb = {t: sum(bool(e.get("pass")) for e in base[t]) / len(base[t]) for t in ids}
    pt = {t: sum(bool(e.get("pass")) for e in treat[t]) / len(treat[t]) for t in ids}
    d = {t: pt[t] - pb[t] for t in ids}
    delta = 100 * sum(d.values()) / len(ids)
    dbs = collections.defaultdict(list)
    for t in ids:
        dbs[tasks[t]["schema"]].append(t)
    db_list = sorted(dbs)
    rng = random.Random(a.seed)
    boots = []
    for _ in range(a.boot):
        sample = [rng.choice(db_list) for _ in db_list]
        ts = [t for db in sample for t in dbs[db]]
        boots.append(100 * sum(d[t] for t in ts) / len(ts))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    # DB-weighted mean (each database counts once)
    dbw = 100 * sum(sum(d[t] for t in dbs[db]) / len(dbs[db]) for db in db_list) / len(db_list)
    # secondary: exact McNemar on per-task seed-count wins/losses; DB sign-flip test on per-DB sums
    wins = sum(1 for t in ids if pt[t] > pb[t]); losses = sum(1 for t in ids if pt[t] < pb[t])
    p_mc = mcnemar_exact(wins, losses)
    db_sums = [sum(d[t] for t in dbs[db]) for db in db_list]
    obs = sum(db_sums)
    flips = 0
    for _ in range(a.boot):
        s = sum(x if rng.random() < 0.5 else -x for x in db_sums)
        flips += abs(s) >= abs(obs) - 1e-12
    p_flip = flips / a.boot
    print(f"\nPRIMARY  Δ = {delta:+.2f} points (mean paired per-task difference) · 95 % DB-cluster bootstrap CI [{lo:+.2f}, {hi:+.2f}] "
          f"({len(db_list)} databases, B={a.boot}) · DB-weighted Δ {dbw:+.2f}")
    print(f"SECONDARY paired tasks: treat better on {wins}, base better on {losses}, tie {len(ids)-wins-losses} · exact McNemar p = {p_mc:.3f} · "
          f"DB sign-flip p = {p_flip:.3f}")
    verdict = "BEATS BASE" if lo > 0 else "not distinguishable from base"
    size = "meaningful (≥ +%.0f)" % a.margin if delta >= a.margin else "below the pre-declared margin"
    print(f"VERDICT: {verdict}; effect {size}")
    # strata
    seen = {s.strip() for s in a.seen_schemas.split(",") if s.strip()}
    for label, sel in (("unseen-schema tasks", [t for t in ids if tasks[t]["schema"] not in seen]),
                       ("seen-schema tasks", [t for t in ids if tasks[t]["schema"] in seen]),
                       ("base-reachable (p̂_base > 0)", [t for t in ids if pb[t] > 0]),
                       ("base-never (p̂_base = 0)", [t for t in ids if pb[t] == 0])):
        if sel:
            print(f"stratum {label:30s} n={len(sel):3d} · base {100*sum(pb[t] for t in sel)/len(sel):5.1f} → treat "
                  f"{100*sum(pt[t] for t in sel)/len(sel):5.1f} · Δ {100*sum(d[t] for t in sel)/len(sel):+.2f}")
    # transitions: per task, compare outcome mixes (seed-count level)
    trans = collections.Counter()
    for t in ids:
        ob = collections.Counter(outcome(e) for e in base[t]); ot = collections.Counter(outcome(e) for e in treat[t])
        nb, nt = len(base[t]), len(treat[t])
        gain = ot["pass"] / nt - ob["pass"] / nb
        if gain > 0:
            src = "nosub" if ob["nosub"] / nb - ot["nosub"] / nt >= ob["wrong"] / nb - ot["wrong"] / nt else "wrong"
            trans[f"{src}→pass"] += gain
        elif gain < 0:
            dst = "nosub" if ot["nosub"] / nt - ob["nosub"] / nb >= ot["wrong"] / nt - ob["wrong"] / nb else "wrong"
            trans[f"pass→{dst}"] += -gain
    print("transitions (task-rate mass):", {k: round(100 * v / len(ids), 2) for k, v in sorted(trans.items())})


if __name__ == "__main__":
    main()
