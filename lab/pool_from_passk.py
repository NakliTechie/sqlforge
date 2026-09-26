"""Assemble the climb-2 training pool from the base pass@8 measurements (leg Experiments 9–10): every task whose base
pass rate under 8 samples is strictly between 0 and 1 (GRPO needs outcome variance inside a group; never-pass tasks carry
no signal and, on BIRD, are often label errors; always-pass tasks are already solved). Each task gets p_hat and a
`set` label; dynamic sampling in the trainer rests tasks that saturate during the climb.

  uv run python -m lab.pool_from_passk --out lab/climb2_pool.json
"""
from __future__ import annotations

import argparse
import collections
import json

SOURCES = [  # (taskfile, pass@8 jsonl, set label)
    ("lab/birdtrain_cand.json", "runs/passk2/birdtrain-base.jsonl", "birdtrain"),
    ("lab/birdtrain_cand2.json", "runs/passk3/birdtrain-base.jsonl", "birdtrain"),
    ("lab/tpcds_tasks.json", "runs/passk2/tpcds-base.jsonl", "tpcds"),
    ("lab/tpch_tasks.json", "runs/passk/tpch-base.jsonl", "tpch"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lab/climb2_pool.json")
    ap.add_argument("--lo", type=float, default=0.0, help="keep p_hat > lo")
    ap.add_argument("--hi", type=float, default=1.0, help="keep p_hat < hi")
    a = ap.parse_args()
    pool = []
    import os
    for tf, jl, label in SOURCES:
        if not os.path.exists(jl):
            print(f"{label:10s} skipped: {jl} not measured yet"); continue
        tasks = json.load(open(tf))
        eps = [json.loads(l) for l in open(jl) if l.strip()]
        c = collections.Counter(e["task"] for e in eps if e.get("pass")); n = collections.Counter(e["task"] for e in eps)
        kept = 0
        for t in tasks:
            if n[t["id"]] == 0:
                continue
            p = c[t["id"]] / n[t["id"]]
            if a.lo < p < a.hi:
                t = dict(t); t["p_hat"] = round(p, 3); t["set"] = label; pool.append(t); kept += 1
        print(f"{label:10s} {kept:4d} of {len(tasks)} kept ({a.lo} < p_hat < {a.hi})")
    json.dump(pool, open(a.out, "w"), indent=1)
    ph = sorted(t["p_hat"] for t in pool)
    print(f"wrote {len(pool)} tasks → {a.out} · p_hat quartiles {ph[len(ph)//4]:.3f} / {ph[len(ph)//2]:.3f} / {ph[3*len(ph)//4]:.3f} · "
          f"databases {len({t['schema'] for t in pool})}")


if __name__ == "__main__":
    main()
