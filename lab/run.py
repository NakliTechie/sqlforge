"""Lockbox runner. Usage: uv run python -m lab.run [--tasks id,id] [--seeds 1,2,3] [--tag name] [--max-turns N]
Writes runs/<tag>.json with per-episode results and prints the summary metrics."""
import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.loop import run_episode
from lab import schemas, spider2, tasks, verify

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
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--taskfile", default="", help="JSON list of tasks (e.g. lab/synth_tasks.json) instead of the fixed set")
    ap.add_argument("--sample", type=int, default=0, help="random sample of K tasks (seeded) from the task set")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS, help="rollout turn cap (episode ends NOSUBMIT at the cap)")
    ap.add_argument("--no-think", dest="think", action="store_false", help="Ollama think=false; default on")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai"],
                    help="ollama = laptop /api/chat; openai = vLLM /v1/chat/completions (GPU)")
    ap.add_argument("--url", default=None, help="server endpoint; default per backend (harness/loop.py URLS)")
    ap.add_argument("--db-dir", default="", help="spider2 tasks: directory holding the .sqlite files (overrides db_path)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="episodes in flight at once; set OLLAMA_NUM_PARALLEL >= this on the server. Batched requests "
                         "are not bit-identical to serial ones, so compare parallel runs with parallel runs")
    a = ap.parse_args()
    ids = [s for s in a.tasks.split(",") if s]
    pool = json.load(open(a.taskfile)) if a.taskfile else tasks.TASKS
    tset = [t for t in pool if not ids or t["id"] in ids]
    if a.sample:
        import random
        tset = random.Random(0).sample(tset, min(a.sample, len(tset)))
    seeds = [int(s) for s in a.seeds.split(",")]
    Path("runs").mkdir(exist_ok=True)
    jl = Path(f"runs/{a.tag}.jsonl")
    eps = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()] if jl.exists() else []
    done = {(e["task"], e["seed"]) for e in eps}
    t0 = time.time()
    lock = threading.Lock()
    tdir = Path(f"runs/{a.tag}")
    tdir.mkdir(exist_ok=True)

    def one(t, seed):
        ts = time.time()
        spider = t.get("kind") == "spider2"
        if spider and a.db_dir:
            t["db_path"] = str(Path(a.db_dir) / f"{t['schema']}.sqlite")
        run_sql = spider2.make_run_sql(t["db_path"]) if spider else make_run_sql(t["schema"])
        ep = run_episode(t["ddl"], t["q"], run_sql, model=a.model, seed=seed,
                         max_turns=a.max_turns, num_ctx=NUM_CTX, think=a.think,
                         backend=a.backend, url=a.url, engine="SQLite" if spider else "DuckDB")
        sql = ep["submitted_sql"]
        if not sql:
            res = {"pass": False, "gates": {}}
        elif spider:
            res = spider2.verify_spider(t, sql)
        else:
            res = verify.verify(t, sql, tasks.HIDDEN_SEEDS)
        row = {"task": t["id"], "hops": t["hops"], "seed": seed, "pass": res["pass"],
               "gate": None if res["pass"] else (verify.first_failed_gate(res) if sql else "NOSUBMIT"),
               "sql": sql, **{k: ep[k] for k in ("via", "turns", "n_run_sql", "hit_cap", "final_text")},
               "secs": round(time.time() - ts, 1)}
        (tdir / f"{t['id']}-s{seed}.json").write_text(json.dumps(
            {"task": t["id"], "q": t["q"], "gold": (t["gold"] or [t.get("gold_sql")])[0], "row": row, "trace": ep["trace"]}, indent=1, default=str))
        with lock:
            eps.append(row)
            with jl.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            print(f"{t['id']:7s} s{seed} {'PASS' if row['pass'] else 'fail@' + str(row['gate']):12s} "
                  f"turns={row['turns']:2d} runs={row['n_run_sql']:2d} {row['secs']}s", flush=True)

    todo = [(t, seed) for t in tset for seed in seeds if (t["id"], seed) not in done]
    with ThreadPoolExecutor(max(1, a.parallel)) as ex:
        list(ex.map(lambda ts_: one(*ts_), todo))  # list() re-raises the first worker exception
    n = len(eps)
    summary = {
        "tag": a.tag, "model": a.model, "max_turns": a.max_turns, "think": a.think, "backend": a.backend, "n_episodes": n, "wall_min": round((time.time() - t0) / 60, 1),
        "exec_acc": round(sum(e["pass"] for e in eps) / n, 4),
        "turn_cap_rate": round(sum(e["hit_cap"] for e in eps) / n, 4),
        "no_submit_rate": round(sum(e["sql"] is None for e in eps) / n, 4),
        "gate_fail_counts": dict(Counter(e["gate"] for e in eps if not e["pass"])),
        "error_count": sum(e.get("via") == "error" for e in eps),
        "exec_acc_by_hops": {h: round(sum(e["pass"] for e in eps if e["hops"] == h) / max(1, sum(1 for e in eps if e["hops"] == h)), 3) for h in sorted({e["hops"] for e in eps})},
        "mean_turns": round(sum(e["turns"] for e in eps) / n, 2),
    }
    per_task, hops_of = {}, {}
    for e in eps:
        per_task.setdefault(e["task"], []).append(e["pass"])
        hops_of[e["task"]] = e["hops"]

    def zone(v):
        return "learnable" if 0 < sum(v) < len(v) else ("saturated" if all(v) else "unsolved")
    multi = {t: v for t, v in per_task.items() if len(v) > 1}
    zones = Counter(zone(v) for v in multi.values())
    summary["zones"] = dict(zones)
    summary["learnable_share"] = round(zones["learnable"] / max(1, sum(zones.values())), 4)
    by_hops = {}
    for h in sorted({hops_of[t] for t in multi}):
        zh = Counter(zone(v) for t, v in multi.items() if hops_of[t] == h)
        by_hops[h] = {"n": sum(zh.values()), **dict(zh), "learnable_share": round(zh["learnable"] / max(1, sum(zh.values())), 3)}
    summary["zones_by_hops"] = by_hops
    Path(f"runs/{a.tag}.json").write_text(json.dumps({"summary": summary, "episodes": eps}, indent=1, default=str))
    print("METRIC exec_acc", summary["exec_acc"])
    print("METRIC learnable_share", summary["learnable_share"])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
