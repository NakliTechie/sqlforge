"""Lockbox self-check: every gold passes its own ladder on all hidden seeds; a hardcoded
answer computed on the visible snapshot fails on hidden seeds; every gold returns >=1 row."""
import sys
from lab import tasks, verify, schemas

bad = 0
for t in tasks.TASKS:
    res = verify.verify(t, t["gold"][0], tasks.HIDDEN_SEEDS)
    if not res["pass"]:
        bad += 1
        print("GOLD FAILS", t["id"], verify.first_failed_gate(res), res["gates"])
        continue
    # visible result → hardcode → must fail on hidden
    con = schemas.build(t["schema"], tasks.VISIBLE_SEED)
    cols, rows = verify.execute(con, t["gold"][0])
    if len(rows) == 0:
        bad += 1; print("EMPTY GOLD on visible", t["id"]); continue
    lits = " UNION ALL ".join("SELECT " + ", ".join(f"{repr(v) if isinstance(v,str) else ('NULL' if v is None else str(v))} AS {c}" for c, v in zip(cols, r)) for r in rows)
    hc = verify.verify(t, lits, tasks.HIDDEN_SEEDS)
    if hc["pass"]:
        bad += 1; print("HARDCODE PASSES (verifier too weak)", t["id"], rows[:3])
    print(f"ok {t['id']:7s} hops={t['hops']} visible_rows={len(rows)} hardcode_fails_at={verify.first_failed_gate(hc)}")
print("BAD" if bad else "ALL OK", bad)
sys.exit(1 if bad else 0)
