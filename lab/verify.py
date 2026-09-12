"""The verifier ladder. Lockbox: nothing in harness/ may import or modify this.

verify(task, sql, seeds) -> dict with per-gate results. Reward is binary: all of G0..G4 pass.

G0 parse    sqlglot, duckdb dialect
G1 bind     sqlglot.optimizer.qualify against the task schema
G2 lint     sqlfluff, diagnostics only (never affects pass/fail)
G3 execute  DuckDB on each hidden snapshot, timeout + row cap
G4 correct  result match vs gold on every hidden snapshot (Spider 2.0 rules)
"""
from __future__ import annotations

import math
import threading
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

from lab import schemas

ROW_CAP = 10_000
TIMEOUT_S = 10.0
ABS_TOL = 1e-2


# ------------------------------------------------------------------ G0 / G1
def _schema_map(ddl: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for stmt in sqlglot.parse(ddl, read="duckdb"):
        if isinstance(stmt, exp.Create) and stmt.this and isinstance(stmt.this, exp.Schema):
            table = stmt.this.this.name
            cols = {}
            for c in stmt.this.expressions:
                if isinstance(c, exp.ColumnDef):
                    cols[c.name] = c.args["kind"].sql("duckdb") if c.args.get("kind") else "UNKNOWN"
            out[table] = cols
    return out


def g0_parse(sql: str) -> tuple[bool, str, exp.Expression | None]:
    try:
        trees = sqlglot.parse(sql, read="duckdb")
    except Exception as e:  # ParseError
        return False, f"{type(e).__name__}: {e}", None
    trees = [t for t in trees if t is not None]
    if len(trees) != 1:
        return False, f"expected exactly 1 statement, got {len(trees)}", None
    if not isinstance(trees[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return False, f"not a query: {type(trees[0]).__name__}", None
    return True, "", trees[0]


def g1_bind(tree: exp.Expression, ddl: str) -> tuple[bool, str]:
    try:
        qualify(tree.copy(), schema=_schema_map(ddl), dialect="duckdb", validate_qualify_columns=True)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, ""


# ------------------------------------------------------------------ G2
def g2_lint(sql: str) -> list[str]:
    try:
        from sqlfluff.core import Linter

        res = Linter(dialect="duckdb").lint_string(sql)
        return [f"{v.rule_code()}: {v.description}" for v in res.get_violations()][:20]
    except Exception as e:
        return [f"lint-error: {type(e).__name__}: {e}"]


# ------------------------------------------------------------------ G3
class ExecError(Exception):
    pass


def execute(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[list[str], list[tuple]]:
    """Run sql with a wall-clock timeout; return (columns, rows) capped at ROW_CAP."""
    result: dict[str, Any] = {}

    def run():
        try:
            cur = con.execute(sql)
            result["cols"] = [d[0] for d in cur.description]
            result["rows"] = cur.fetchmany(ROW_CAP + 1)
        except Exception as e:  # noqa: BLE001
            result["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(TIMEOUT_S)
    if t.is_alive():
        con.interrupt()
        t.join(2)
        raise ExecError(f"timeout after {TIMEOUT_S}s")
    if "err" in result:
        raise ExecError(result["err"])
    rows = result["rows"]
    if len(rows) > ROW_CAP:
        raise ExecError(f"more than {ROW_CAP} rows")
    return result["cols"], rows


# ------------------------------------------------------------------ G4 (Spider 2.0 rules)
def _norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    try:
        from decimal import Decimal

        if isinstance(v, Decimal):
            return float(v)
    except Exception:
        pass
    return str(v)


def _cell_eq(a, b) -> bool:
    a, b = _norm(a), _norm(b)
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, abs_tol=ABS_TOL)
    return a == b


def _vec_match(g: list, p: list, ignore_order: bool) -> bool:
    if len(g) != len(p):
        return False
    if ignore_order:
        g = sorted(g, key=lambda x: (x is None, str(_norm(x))))
        p = sorted(p, key=lambda x: (x is None, str(_norm(x))))
    return all(_cell_eq(x, y) for x, y in zip(g, p))


def compare(gold_cols: list[str], gold_rows: list[tuple], pred_cols: list[str], pred_rows: list[tuple],
            condition_cols: list[int] | None, ignore_order: bool) -> tuple[bool, str]:
    """Spider 2.0 compare_pandas_table: each gold column named in condition_cols (default: all)
    must appear as some column vector in the prediction, row counts must agree."""
    if len(gold_rows) != len(pred_rows):
        return False, f"row count gold={len(gold_rows)} pred={len(pred_rows)}"
    idx = condition_cols if condition_cols else list(range(len(gold_cols)))
    pred_vectors = [[r[j] for r in pred_rows] for j in range(len(pred_cols))]
    for j in idx:
        gvec = [r[j] for r in gold_rows]
        if not any(_vec_match(gvec, pv, ignore_order) for pv in pred_vectors):
            return False, f"gold column '{gold_cols[j]}' not matched by any predicted column"
    return True, ""


# ------------------------------------------------------------------ the ladder
def verify(task: dict, sql: str, hidden_seeds: list[int]) -> dict:
    """task keys: schema, ddl, gold (list of sql), condition_cols, ignore_order."""
    out: dict[str, Any] = {"gates": {}, "pass": False}
    ok, msg, tree = g0_parse(sql)
    out["gates"]["G0"] = {"pass": ok, "msg": msg}
    if not ok:
        return out
    ok, msg = g1_bind(tree, task["ddl"])
    out["gates"]["G1"] = {"pass": ok, "msg": msg}
    if not ok:
        return out
    out["gates"]["G2"] = {"pass": True, "diag": g2_lint(sql)}

    g3_ok, g4_ok, g3_msg, g4_msg = True, True, "", ""
    for seed in hidden_seeds:
        con = schemas.build(task["schema"], seed)
        try:
            pcols, prows = execute(con, sql)
        except ExecError as e:
            g3_ok, g3_msg = False, f"seed {seed}: {e}"
            break
        matched = False
        for gold in task["gold"]:
            gcols, grows = execute(con, gold)
            m, why = compare(gcols, grows, pcols, prows, task.get("condition_cols"), task.get("ignore_order", False))
            if m:
                matched = True
                break
        if not matched:
            g4_ok, g4_msg = False, f"seed {seed}: {why}"
            break
    out["gates"]["G3"] = {"pass": g3_ok, "msg": g3_msg}
    if not g3_ok:
        return out
    out["gates"]["G4"] = {"pass": g4_ok, "msg": g4_msg}
    out["pass"] = g4_ok
    return out


def first_failed_gate(res: dict) -> str | None:
    for g in ("G0", "G1", "G3", "G4"):
        r = res["gates"].get(g)
        if r is None:
            return g
        if not r["pass"]:
            return g
    return None
