"""Spider 2.0-Lite, SQLite slice (135 tasks over 30 local databases) as sqlforge's outside held-out evaluation.

Eval-only: the repo is MIT but its maintainers advise against fine-tuning on the gold, and there are no hidden
snapshots — each task has ONE fixed database, so the verifier compares the submitted SQL's result against the gold
result rows (Spider 2.0's own rules: condition_cols, ignore_order, abs_tol 1e-2, any of several golds), on that
database. The harness runs the model's SQL through sqlite3 (the benchmark's dialect); the system prompt names SQLite.

Build:  uv run python -m lab.spider2 --repo data/spider2-lite/repo --db-dir data/spider2-lite/localdb \\
            --out lab/spider2_sqlite.json
Check:  uv run python -m lab.spider2 --check lab/spider2_sqlite.json      # gold SQL (where given) must pass
Run:    uv run python -m lab.run --taskfile lab/spider2_sqlite.json --backend openai --model ... --seeds 1 ...
        (lab.run detects kind == "spider2" and uses run_sql / verify from this module; --db-dir overrides db paths)
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

from lab import verify

QUERY_TIMEOUT_S = 60
MAX_ROWS = 5000


def schema_ddl(db_path: str) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL "
                           "AND name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()
    finally:
        con.close()
    return "\n\n".join(r[0].strip().rstrip(";") + ";" for r in rows)


def make_run_sql(db_path: str):
    """The harness tool: read-only sqlite3 with a statement timeout via the progress handler."""
    def run_sql(sql: str):
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        deadline = [0]
        import time
        t0 = time.time()
        con.set_progress_handler(lambda: 1 if time.time() - t0 > QUERY_TIMEOUT_S else 0, 10_000)
        try:
            cur = con.execute(sql)
            rows = cur.fetchmany(MAX_ROWS + 1)
            cols = [d[0] for d in cur.description] if cur.description else []
            return cols, rows[:MAX_ROWS], None
        except Exception as e:  # sqlite3.OperationalError, Warning (multiple statements), interrupted
            return [], [], f"{type(e).__name__}: {e}"
        finally:
            con.close()
    return run_sql


def verify_spider(task: dict, sql: str) -> dict:
    """G0/G1 are skipped (SQLite dialect, sqlglot's qualify is unreliable on these schemas); G3 execute on the one
    database; G4 result match against any gold result set under the task's condition_cols / ignore_order."""
    out = {"gates": {"G0": {"pass": True, "skipped": True}, "G1": {"pass": True, "skipped": True}}, "pass": False}
    if not sql or not sql.strip():
        out["gates"]["G3"] = {"pass": False, "msg": "empty SQL"}
        return out
    cols, rows, err = make_run_sql(task["db_path"])(sql)
    if err:
        out["gates"]["G3"] = {"pass": False, "msg": err[:300]}
        return out
    out["gates"]["G3"] = {"pass": True}
    rows = [tuple(0 if v is None else v for v in r) for r in rows]  # official normalize(): NaN/NULL → 0
    if task.get("match") == "bird_ex":  # BIRD EX: set(pred rows) == set(gold rows); column order and duplicates matter
        def key(r):
            return tuple(verify._norm(v) for v in r)
        gold = task["gold_results"][0]
        ok = sorted(map(key, [tuple(0 if v is None else v for v in r) for r in gold["rows"]]), key=str) == sorted(map(key, rows), key=str)
        out["gates"]["G4"] = {"pass": ok} if ok else {"pass": False, "msg": f"row set differs: gold {len(gold['rows'])} rows, pred {len(rows)} rows"}
        out["pass"] = ok
        return out
    for i, gold in enumerate(task["gold_results"]):
        cc = task["condition_cols"]
        if cc and isinstance(cc[0], list):  # per-gold condition cols
            cc = cc[i] if i < len(cc) else cc[-1]
        ok, msg = verify.compare(gold["cols"], [tuple(0 if v in ("", None) else v for v in r) for r in gold["rows"]], cols, rows,
                                 cc or None, task["ignore_order"])
        if ok:
            out["gates"]["G4"] = {"pass": True, "gold": i}
            out["pass"] = True
            return out
        out["gates"]["G4"] = {"pass": False, "msg": msg}
    return out


def build(repo: Path, db_dir: Path) -> list[dict]:
    lite = repo / "spider2-lite"
    ev = {}
    for line in (lite / "evaluation_suite/gold/spider2lite_eval.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            ev[d["instance_id"]] = d
    docs_dir = next((p for p in lite.glob("resource/*") if p.is_dir() and any(p.glob("*.md"))), None)
    res_dir = lite / "evaluation_suite/gold/exec_result"
    sql_dir = lite / "evaluation_suite/gold/sql"
    tasks = []
    for line in (lite / "spider2-lite.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r["instance_id"]
        if not iid.startswith("local"):
            continue
        golds = []
        for f in sorted(res_dir.glob(f"{iid}.csv")) + sorted(res_dir.glob(f"{iid}_*.csv")):
            rows = list(csv.reader(open(f, newline="")))
            golds.append({"file": f.name, "cols": rows[0], "rows": rows[1:]})
        doc = ""
        if r.get("external_knowledge") and docs_dir and (docs_dir / r["external_knowledge"]).exists():
            doc = (docs_dir / r["external_knowledge"]).read_text()
        db_path = db_dir / f"{r['db']}.sqlite"
        q = r["question"] + (f"\n\nReference document ({r['external_knowledge']}):\n{doc}" if doc else "")
        tasks.append({"kind": "spider2", "id": iid, "schema": r["db"], "db_path": str(db_path), "hops": 0,
                      "q": q, "ddl": schema_ddl(str(db_path)), "gold": [], "gold_sql": (sql_dir / f"{iid}.sql").read_text().strip()
                      if (sql_dir / f"{iid}.sql").exists() else None, "gold_results": golds,
                      "condition_cols": ev[iid].get("condition_cols") or [], "ignore_order": bool(ev[iid].get("ignore_order")),
                      "external_knowledge": r.get("external_knowledge")})
    return tasks


def build_bird(json_path: Path, db_root: Path) -> list[dict]:
    """BIRD Mini-Dev (500 SELECT-only SQLite questions from BIRD dev). Gold results are the gold SQL's rows on the one
    database; match = BIRD execution accuracy (exact row set). The evidence hint is appended to the question, as in
    BIRD's own prompts."""
    items = json.loads(json_path.read_text())
    assert {"db_id", "question", "SQL"} <= set(items[0]), sorted(items[0])
    tasks, skipped = [], []
    for k, it in enumerate(items):
        db = it["db_id"]
        db_path = next(iter(db_root.glob(f"**/{db}/{db}.sqlite")), None) or next(iter(db_root.glob(f"**/{db}.sqlite")), None)
        if db_path is None:
            skipped.append((db, "no db file")); continue
        cols, rows, err = make_run_sql(str(db_path))(it["SQL"])
        if err:
            skipped.append((it.get("question_id", k), err[:80])); continue
        q = it["question"] + (f"\n\nEvidence: {it['evidence']}" if it.get("evidence") else "")
        tasks.append({"kind": "spider2", "match": "bird_ex", "id": f"bird{it.get('question_id', k):04d}", "schema": db,
                      "db_path": str(db_path), "hops": {"simple": 1, "moderate": 2, "challenging": 3}.get(it.get("difficulty"), 0),
                      "difficulty": it.get("difficulty"), "q": q, "ddl": schema_ddl(str(db_path)), "gold": [], "gold_sql": it["SQL"],
                      "gold_results": [{"file": None, "cols": cols, "rows": [list(r) for r in rows]}],
                      "condition_cols": [], "ignore_order": True, "external_knowledge": None})
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-json", help="BIRD Mini-Dev JSON (mini_dev_sqlite.json); builds a BIRD task file instead")
    ap.add_argument("--repo", default="data/spider2-lite/repo")
    ap.add_argument("--db-dir", default="data/spider2-lite/localdb")
    ap.add_argument("--out", default="lab/spider2_sqlite.json")
    ap.add_argument("--check", metavar="TASKFILE", help="run each task's gold SQL (where given) through verify_spider")
    a = ap.parse_args()
    if a.check:
        tasks = json.load(open(a.check))
        with_sql = [t for t in tasks if t.get("gold_sql")]
        ok = 0
        for t in with_sql:
            res = verify_spider(t, t["gold_sql"])
            ok += res["pass"]
            if not res["pass"]:
                print("FAIL", t["id"], json.dumps(res["gates"])[:200])
        print(f"gold SQL passes verify_spider: {ok}/{len(with_sql)} (of {len(tasks)} tasks)")
        sys.exit(0 if ok == len(with_sql) else 1)
    tasks = build_bird(Path(a.bird_json), Path(a.db_dir)) if a.bird_json else build(Path(a.repo), Path(a.db_dir))
    json.dump(tasks, open(a.out, "w"), indent=1)
    if a.bird_json:
        import collections
        print(f"wrote {len(tasks)} BIRD tasks → {a.out} · difficulty {dict(collections.Counter(t['difficulty'] for t in tasks))} · "
              f"dbs {len({t['schema'] for t in tasks})} · DDL chars median {sorted(len(t['ddl']) for t in tasks)[len(tasks)//2]}")
        return
    docs = sum(1 for t in tasks if t["external_knowledge"])
    docs_in = sum(1 for t in tasks if t["external_knowledge"] and "Reference document" in t["q"])
    print(f"wrote {len(tasks)} tasks → {a.out} · golds/task min {min(len(t['gold_results']) for t in tasks)} max "
          f"{max(len(t['gold_results']) for t in tasks)} · with gold SQL {sum(1 for t in tasks if t['gold_sql'])} · "
          f"external docs {docs_in}/{docs} · DDL chars median {sorted(len(t['ddl']) for t in tasks)[len(tasks)//2]}")


if __name__ == "__main__":
    main()
