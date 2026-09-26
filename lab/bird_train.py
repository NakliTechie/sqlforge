"""BIRD train (9,428 questions, 69 SQLite DBs; disjoint from the dev/Mini-Dev DBs) as the real-schema climb-2 pool source.
Train has no difficulty labels, so a proxy from the gold SQL picks the moderate/challenging-like slice; the 8-seed base pass
on GCP then keeps 0.1 <= p̂ <= 0.7. Tasks use lab/spider2.build_bird's shape (kind spider2, match bird_ex, evidence appended).

Extract: unzip data/bird-train/train.zip -d data/bird-train && unzip data/bird-train/train/train_databases.zip -d data/bird-train/train
Build:   uv run python -m lab.bird_train --out lab/birdtrain_cand.json --n 600
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from lab.spider2 import build_bird


def difficulty_proxy(sql: str) -> dict:
    s = " " + re.sub(r"\s+", " ", sql.strip().rstrip(";")).upper() + " "
    f = {"joins": s.count(" JOIN "), "subq": max(0, s.count("(SELECT") + s.count("( SELECT")),
         "group": int(" GROUP BY " in s), "having": int(" HAVING " in s), "order": int(" ORDER BY " in s),
         "case": s.count(" CASE "), "window": int(" OVER (" in s or " OVER(" in s), "distinct": int(" DISTINCT " in s),
         "arith": len(re.findall(r"[\w)]\s*[*/]\s*[\w(]", s)), "cast": s.count("CAST("), "len": len(s),
         "union": int(" UNION " in s or " EXCEPT " in s or " INTERSECT " in s)}
    f["score"] = (2 * f["joins"] + 3 * f["subq"] + f["group"] + 2 * f["having"] + f["case"] + 3 * f["window"] + f["arith"]
                  + 2 * f["union"] + (1 if f["len"] > 200 else 0) + (1 if f["len"] > 350 else 0))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/bird-train/train/train.json")
    ap.add_argument("--db-root", default="data/bird-train/train/train_databases")
    ap.add_argument("--out", default="lab/birdtrain_cand.json")
    ap.add_argument("--n", type=int, default=600, help="candidate count for the pass@8 measurement")
    ap.add_argument("--min-score", type=int, default=4, help="proxy score floor (moderate/challenging-like)")
    ap.add_argument("--max-db-mb", type=float, default=400, help="skip databases larger than this (bucket sync cost)")
    ap.add_argument("--per-db", type=int, default=20, help="cap per database, for schema diversity")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    items = json.load(open(a.json))
    sizes = {p.parent.name: p.stat().st_size / 1e6 for p in Path(a.db_root).glob("*/*.sqlite")}
    print(f"{len(items)} questions · {len(sizes)} databases · {sum(1 for v in sizes.values() if v > a.max_db_mb)} over {a.max_db_mb} MB skipped")
    for k, it in enumerate(items):
        it["question_id"] = it.get("question_id", k)
        it["proxy"] = difficulty_proxy(it["SQL"])
    keep = [it for it in items if it["proxy"]["score"] >= a.min_score and sizes.get(it["db_id"], 1e9) <= a.max_db_mb]
    print(f"score >= {a.min_score}: {len(keep)} of {len(items)} · score histogram all: "
          f"{dict(sorted(collections.Counter(min(it['proxy']['score'], 12) for it in items).items()))}")
    random.seed(a.seed); random.shuffle(keep)
    per, chosen = collections.Counter(), []
    for it in sorted(keep, key=lambda x: -x["proxy"]["score"]):  # hardest first, capped per DB
        if per[it["db_id"]] < a.per_db:
            per[it["db_id"]] += 1; chosen.append(it)
        if len(chosen) >= a.n:
            break
    tmp = Path(a.out).with_suffix(".src.json"); json.dump(chosen, open(tmp, "w"))
    tasks = build_bird(tmp, Path(a.db_root))
    tmp.unlink()
    for t in tasks:
        t["id"] = "bt" + t["id"][4:]; t["hops"] = 4
        src = next(it for it in chosen if it["question_id"] == int(t["id"][2:]))
        t["proxy_score"] = src["proxy"]["score"]
    json.dump(tasks, open(a.out, "w"), indent=1)
    dbs = collections.Counter(t["schema"] for t in tasks)
    print(f"wrote {len(tasks)} tasks → {a.out} · {len(dbs)} databases (max per db {max(dbs.values())}) · "
          f"total db MB to sync {sum(sizes[d] for d in dbs):.0f} · proxy score median {sorted(t['proxy_score'] for t in tasks)[len(tasks)//2]}")


if __name__ == "__main__":
    main()
