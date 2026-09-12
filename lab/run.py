"""Lockbox runner. Usage: uv run python -m lab.run [--tasks id,id] [--seeds 1,2,3] [--tag name]
Writes runs/<tag>.json with per-episode results and prints the summary metrics."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from harness.loop import run_episode
from lab import schemas, tasks, verify

MODEL = "qwen3.5:4b"
MAX_TURNS = 40
NUM_CTX = 32768
SEEDS = [1, 2, 3]


def make_run_sql(schema):
    con = schemas.build(schema, tasks.VISIBLE_SEED)

    def run_sql(sql):
        try:
            cols, rows = verify.execute(con, sql)
            return cols, rows, None
        except verify.ExecError as e:
            return [], [], str(e)
    return run_sql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    a = ap.parse_args()
    ids = [s for s in a.tasks.split(",") if s]
    tset = [t for t in tasks.TASKS if not ids or t["id"] in ids]
    seeds = [int(s) for s in a.seeds.split(",")]
    eps = []
    t0 = time.time()
    for t in tset:
        for seed in seeds:
            ts = time.time()
            ep = run_episode(t["ddl"], t["q"], make_run_sql(t["schema"]), model=MODEL, seed=seed,
                             max_turns=MAX_TURNS, num_ctx=NUM_CTX)
            sql = ep["submitted_sql"]
            res = verify.verify(t, sql, tasks.HIDDEN_SEEDS) if sql else {"pass": False, "gates": {}}
            row = {"task": t["id"], "hops": t["hops"], "seed": seed, "pass": res["pass"],
                   "gate": None if res["pass"] else (verify.first_failed_gate(res) if sql else "NOSUBMIT"),
                   "sql": sql, **{k: ep[k] for k in ("via", "turns", "n_run_sql", "hit_cap", "final_text")},
                   "secs": round(time.time() - ts, 1)}
            eps.append(row)
            print(f"{t['id']:7s} s{seed} {'PASS' if row['pass'] else 'fail@' + str(row['gate']):12s} turns={row['turns']:2d} runs={row['n_run_sql']:2d} {row['secs']}s", flush=True)
    n = len(eps)
    summary = {
        "tag": a.tag, "model": MODEL, "n_episodes": n, "wall_min": round((time.time() - t0) / 60, 1),
        "exec_acc": round(sum(e["pass"] for e in eps) / n, 4),
        "turn_cap_rate": round(sum(e["hit_cap"] for e in eps) / n, 4),
        "no_submit_rate": round(sum(e["sql"] is None for e in eps) / n, 4),
        "gate_fail_counts": dict(Counter(e["gate"] for e in eps if not e["pass"])),
        "exec_acc_by_hops": {h: round(sum(e["pass"] for e in eps if e["hops"] == h) / max(1, sum(1 for e in eps if e["hops"] == h)), 3) for h in sorted({e["hops"] for e in eps})},
        "mean_turns": round(sum(e["turns"] for e in eps) / n, 2),
    }
    Path("runs").mkdir(exist_ok=True)
    Path(f"runs/{a.tag}.json").write_text(json.dumps({"summary": summary, "episodes": eps}, indent=1, default=str))
    print("METRIC exec_acc", summary["exec_acc"])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
