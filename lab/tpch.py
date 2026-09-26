"""TPC-H (scale 0.1) in SQLite as a trainable analytical pool candidate: 8 tables, the 22 business questions with their
official parameters plus 2 parameter variants for 13 of them (48 tasks). Gold SQL is DuckDB's tpch_queries() text with
literals substituted; gold results come from DuckDB on the same dbgen data that is exported to the SQLite file the model
queries. Tasks use the spider2 task shape (kind spider2, SQLite engine, verify_spider with Spider rules on all columns).

Build:  uv run python -m lab.tpch --out lab/tpch_tasks.json --db data/tpch/tpch_sf0.1.sqlite
Check:  uv run python -m lab.tpch --check lab/tpch_tasks.json   # two hand-translated SQLite golds must pass verify_spider
"""
from __future__ import annotations

import argparse
import datetime as dt
import decimal
import json
import os
import re
from pathlib import Path

DOC = """TPC-H data dictionary (scale factor 0.1). Tables: region(r_regionkey, r_name, r_comment); nation(n_nationkey,
n_name, n_regionkey, n_comment); supplier(s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment);
part(p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container, p_retailprice, p_comment); partsupp(ps_partkey,
ps_suppkey, ps_availqty, ps_supplycost, ps_comment); customer(c_custkey, c_name, c_address, c_nationkey, c_phone,
c_acctbal, c_mktsegment, c_comment); orders(o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate,
o_orderpriority, o_clerk, o_shippriority, o_comment); lineitem(l_orderkey, l_partkey, l_suppkey, l_linenumber,
l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate, l_commitdate, l_receiptdate,
l_shipinstruct, l_shipmode, l_comment). Dates are TEXT 'YYYY-MM-DD'. Revenue of a line item = l_extendedprice * (1 -
l_discount). Discounts are fractions (0.06 = 6%). A customer's nation is via c_nationkey; a supplier's via s_nationkey;
a nation's region via n_regionkey. Order priority values look like '1-URGENT'; c_phone starts with a 2-digit country code."""

# per query: NL question template, parameter fields with the DEFAULT literal (as it appears in DuckDB's query text) and
# two variants. Literal substitution is textual on the gold SQL, so each default literal must be unique in that query.
Q = {
 1: ("Pricing summary report: for line items shipped on or before {d} (that is, 90 days before 1998-12-01 for the default), "
     "group by return flag and line status and report the sum of quantity, sum of extended price, sum of discounted price "
     "(extendedprice*(1-discount)), sum of charge (discounted price*(1+tax)), average quantity, average extended price, "
     "average discount and the count of line items, ordered by return flag then line status.",
     {"d": ("CAST('1998-09-02' AS date)", ["CAST('1998-08-15' AS date)", "CAST('1998-10-01' AS date)"])}),
 2: ("Minimum cost supplier: for parts of size 15 whose type ends in 'BRASS', find in the EUROPE region the supplier(s) "
     "offering the minimum supply cost for each such part. Return s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, "
     "s_phone, s_comment ordered by s_acctbal desc, n_name, s_name, p_partkey; first 100 rows.", {}),
 3: ("Shipping priority: for customers in the {seg} market segment, orders placed before {d} with line items shipped after "
     "{d}, return the order key, total revenue (sum of extendedprice*(1-discount)), order date and ship priority per order, "
     "ordered by revenue desc then order date; first 10.",
     {"seg": ("'BUILDING'", ["'AUTOMOBILE'", "'MACHINERY'"]), "d": ("CAST('1995-03-15' AS date)", ["CAST('1995-03-01' AS date)", "CAST('1995-03-28' AS date)"])}),
 4: ("Order priority checking: count orders per order priority for orders placed in the quarter starting {d} (three "
     "months) where at least one line item was received later than its commit date. Order by priority.",
     {"d": ("CAST('1993-07-01' AS date)", ["CAST('1993-10-01' AS date)", "CAST('1994-01-01' AS date)"]), "d2": ("CAST('1993-10-01' AS date)", ["CAST('1994-01-01' AS date)", "CAST('1994-04-01' AS date)"])}),
 5: ("Local supplier volume: for the {r} region, total revenue (sum of extendedprice*(1-discount)) per nation from line "
     "items where the customer and the supplier are in the same nation, for orders placed in the year starting {d}. "
     "Order by revenue desc.",
     {"r": ("'ASIA'", ["'EUROPE'", "'AMERICA'"]), "d": ("CAST('1994-01-01' AS date)", ["CAST('1995-01-01' AS date)", "CAST('1996-01-01' AS date)"]), "d2": ("CAST('1995-01-01' AS date)", ["CAST('1996-01-01' AS date)", "CAST('1997-01-01' AS date)"])}),
 6: ("Forecasting revenue change: total revenue increase (sum of extendedprice*discount) from line items shipped in the "
     "year starting {d} with discount between {lo} and {hi} inclusive and quantity below {qty}.",
     {"d": ("CAST('1994-01-01' AS date)", ["CAST('1995-01-01' AS date)", "CAST('1996-01-01' AS date)"]), "d2": ("CAST('1995-01-01' AS date)", ["CAST('1996-01-01' AS date)", "CAST('1997-01-01' AS date)"]),
      "lo": ("0.05", ["0.07", "0.03"]), "hi": ("0.07", ["0.09", "0.05"]), "qty": ("24", ["25", "23"])}),
 7: ("Volume shipping: revenue (sum of extendedprice*(1-discount)) by supplier nation, customer nation and ship year "
     "for line items shipped in 1995 or 1996 between {a} and {b} (both directions). Order by supplier nation, customer "
     "nation, year.",
     {"a": ("'FRANCE'", ["'JAPAN'", "'BRAZIL'"]), "b": ("'GERMANY'", ["'INDIA'", "'ARGENTINA'"])}),
 8: ("National market share: for each order year, the share of revenue (extendedprice*(1-discount)) from suppliers of "
     "BRAZIL among all orders in the AMERICA region for parts of type 'ECONOMY ANODIZED STEEL', orders placed 1995-01-01 to "
     "1996-12-31. Order by year.", {}),
 9: ("Product type profit measure: for parts whose name contains 'green', profit per nation and order year, where profit = "
     "sum(extendedprice*(1-discount) - supplycost*quantity) by supplier nation. Order by nation, year desc.", {}),
 10: ("Returned item reporting: for orders placed in the quarter starting {d}, the customers with returned line items "
      "(returnflag 'R'): custkey, name, revenue lost (sum of extendedprice*(1-discount)), acctbal, nation, address, "
      "phone, comment, ordered by revenue desc; first 20.",
      {"d": ("CAST('1993-10-01' AS date)", ["CAST('1994-01-01' AS date)", "CAST('1993-07-01' AS date)"]), "d2": ("CAST('1994-01-01' AS date)", ["CAST('1994-04-01' AS date)", "CAST('1993-10-01' AS date)"])}),
 11: ("Important stock identification: for suppliers in GERMANY, part keys whose total stock value (sum of supplycost*"
      "availqty) exceeds 0.0001 of the total stock value of all German suppliers' parts. Order by value desc.", {}),
 12: ("Shipping modes and order priority: for line items with ship mode {m1} or {m2}, commit date before receipt date, "
      "ship date before commit date, received in the year starting {d}: per ship mode, the count of high-priority orders "
      "(priority '1-URGENT' or '2-HIGH') and the count of the others. Order by ship mode.",
      {"m1": ("'MAIL'", ["'AIR'", "'RAIL'"]), "m2": ("'SHIP'", ["'TRUCK'", "'FOB'"]), "d": ("CAST('1994-01-01' AS date)", ["CAST('1995-01-01' AS date)", "CAST('1996-01-01' AS date)"]), "d2": ("CAST('1995-01-01' AS date)", ["CAST('1996-01-01' AS date)", "CAST('1997-01-01' AS date)"])}),
 13: ("Customer distribution: count customers by their number of orders, excluding orders whose comment matches "
      "'%special%requests%'; customers with no orders count as 0. Order by customer count desc then order count desc.", {}),
 14: ("Promotion effect: percentage of revenue (extendedprice*(1-discount)) from parts whose type starts with 'PROMO' "
      "among line items shipped in the month starting {d} (100 * promo revenue / total revenue).",
      {"d": ("CAST('1995-09-01' AS date)", ["CAST('1995-10-01' AS date)", "CAST('1996-03-01' AS date)"]), "d2": ("CAST('1995-10-01' AS date)", ["CAST('1995-11-01' AS date)", "CAST('1996-04-01' AS date)"])}),
 15: ("Top supplier: the supplier(s) with the maximum total revenue (sum of extendedprice*(1-discount)) over line items "
      "shipped in the quarter starting 1996-01-01: suppkey, name, address, phone, total revenue, ordered by suppkey.", {}),
 16: ("Parts/supplier relationship: count of distinct suppliers per (brand, type, size) for parts not of Brand#45, type not "
      "starting with 'MEDIUM POLISHED', size in (49,14,23,45,19,3,36,9), excluding suppliers whose comment matches "
      "'%Customer%Complaints%'. Order by supplier count desc, brand, type, size.", {}),
 17: ("Small-quantity-order revenue: average yearly revenue loss (sum of extendedprice / 7) from line items of parts with "
      "brand {b} and container {c} whose quantity is below 20% of the average quantity ordered for that part.",
      {"b": ("'Brand#23'", ["'Brand#31'", "'Brand#12'"]), "c": ("'MED BOX'", ["'SM CASE'", "'LG PKG'"])}),
 18: ("Large volume customers: customers with orders whose total line quantity exceeds {q}: name, custkey, orderkey, "
      "order date, total price and the summed quantity, ordered by total price desc then order date; first 100.",
      {"q": ("300", ["312", "295"])}),
 19: ("Discounted revenue: total revenue (extendedprice*(1-discount)) for air-shipped ('AIR' or 'AIR REG'), 'DELIVER IN "
      "PERSON' line items matching one of three part/quantity profiles: Brand#12, small containers (SM CASE/BOX/PACK/PKG), "
      "quantity 1-11, size 1-5; Brand#23, medium bags/boxes/packs/packages (MED BAG/BOX/PKG/PACK), quantity 10-20, size "
      "1-10; Brand#34, large containers (LG CASE/BOX/PACK/PKG), quantity 20-30, size 1-15.", {}),
 20: ("Potential part promotion: suppliers in {n} that have an excess of parts whose name starts with {p}: excess = "
      "available quantity above 50% of the quantity shipped of that part by that supplier in the year starting {d}. "
      "Return supplier name and address ordered by name.",
      {"n": ("'CANADA'", ["'PERU'", "'ETHIOPIA'"]), "p": ("'forest%'", ["'sky%'", "'lime%'"]), "d": ("CAST('1994-01-01' AS date)", ["CAST('1995-01-01' AS date)", "CAST('1993-01-01' AS date)"]), "d2": ("CAST('1995-01-01' AS date)", ["CAST('1996-01-01' AS date)", "CAST('1994-01-01' AS date)"])}),
 21: ("Suppliers who kept orders waiting: for suppliers in {n}, count the multi-supplier orders with status 'F' where this "
      "supplier was the only one whose line item was received after its commit date. Order by count desc, name; first 100.",
      {"n": ("'SAUDI ARABIA'", ["'MOROCCO'", "'VIETNAM'"])}),
 22: ("Global sales opportunity: for customers whose phone country code (first two characters) is in 13, 31, 23, 29, 30, 18, "
      "17, with account balance above the average positive balance of such customers, and no orders: per country code the "
      "customer count and total account balance. Order by country code.", {}),
}
IDX_STMT = {}  # queries where DuckDB's text has the default literal more than once are handled by replace-all


def export_sqlite(con, db: Path):
    import sqlite3
    if db.exists():
        db.unlink()
    out = sqlite3.connect(str(db))
    for (t,) in con.execute("select table_name from duckdb_tables() order by 1").fetchall():
        cols = con.execute(f"describe {t}").fetchall()
        def sqlt(ty):
            ty = ty.upper()
            return "INTEGER" if ty in ("BIGINT", "INTEGER") else "REAL" if ty.startswith("DECIMAL") or ty == "DOUBLE" else "TEXT"
        out.execute(f"CREATE TABLE {t} ({', '.join(f'{c[0]} {sqlt(c[1])}' for c in cols)})")
        rows = con.execute(f"select * from {t}").fetchall()
        rows = [tuple(norm(v) for v in r) for r in rows]
        out.executemany(f"INSERT INTO {t} VALUES ({', '.join('?' * len(cols))})", rows)
    out.commit(); out.close()


def norm(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.strftime("%Y-%m-%d")
    return v


def build(db: Path) -> list[dict]:
    import duckdb
    from lab.spider2 import schema_ddl
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.1)")
    export_sqlite(con, db)
    ddl = schema_ddl(str(db))
    qs = dict(con.execute("select query_nr, query from tpch_queries()").fetchall())
    tasks = []
    for nr, (tmpl, params) in Q.items():
        sql0 = qs[nr]
        # DuckDB writes some date literals as CAST('X' AS date) and some as date 'X'; match the form the query uses
        def form(lit):
            m = re.match(r"CAST\('([\d-]+)' AS date\)", lit)
            if m and lit not in sql0:
                alt = f"date '{m.group(1)}'"
                assert alt in sql0, (nr, lit, sql0[:400])
                return alt
            assert lit in sql0, (nr, lit, sql0[:400])
            return lit
        params = {k: (form(d), [form(v) if False else (re.sub(r"CAST\('([\d-]+)' AS date\)", r"date '\1'", v) if form(d).startswith("date ") else v) for v in vs]) for k, (d, vs) in params.items()}
        variants = [{k: d for k, (d, _) in params.items()}]
        if params:
            for i in range(2):
                variants.append({k: v[i] for k, (_, v) in params.items()})
        for vi, vals in enumerate(variants):
            sql = sql0
            for i, k in enumerate(vals):          # two passes: literals → tokens → values (a moved start date must not
                sql = sql.replace(params[k][0], f"@@{i}@@")   # be re-replaced when it equals another param's default)
            for i, k in enumerate(vals):
                sql = sql.replace(f"@@{i}@@", vals[k])
            def show(v):  # literal → what the question shows
                m = re.match(r"(?:CAST\('([\d-]+)' AS date\)|date '([\d-]+)')", v)
                return (m.group(1) or m.group(2)) if m else v.strip("'")
            q = tmpl.format(**{k: show(v) for k, v in vals.items()})
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [[norm(v) for v in r] for r in cur.fetchall()]
            if not rows or all(v in (0, 0.0, None) for r in rows for v in r):
                print(f"drop tpch-q{nr:02d}-v{vi}: degenerate gold ({len(rows)} rows)"); continue
            tasks.append({"kind": "spider2", "id": f"tpch-q{nr:02d}-v{vi}", "schema": "tpch", "db_path": str(db), "hops": 4,
                          "q": q + "\n\nReference document (tpch data dictionary):\n" + DOC, "ddl": ddl, "gold": [],
                          "gold_sql": None, "gold_sql_duckdb": sql, "gold_results": [{"file": None, "cols": cols, "rows": rows}],
                          "condition_cols": [], "ignore_order": "order by" not in sql.lower(), "external_knowledge": "tpch"})
    return tasks


# two hand-translated SQLite golds, to prove the export + compare path end to end
SQLITE_CHECKS = {
 "tpch-q06-v0": "select sum(l_extendedprice * l_discount) as revenue from lineitem where l_shipdate >= '1994-01-01' and "
                "l_shipdate < '1995-01-01' and l_discount between 0.05 and 0.07 and l_quantity < 24",
 "tpch-q01-v0": "select l_returnflag, l_linestatus, sum(l_quantity), sum(l_extendedprice), sum(l_extendedprice*(1-l_discount)), "
                "sum(l_extendedprice*(1-l_discount)*(1+l_tax)), avg(l_quantity), avg(l_extendedprice), avg(l_discount), count(*) "
                "from lineitem where l_shipdate <= '1998-09-02' group by l_returnflag, l_linestatus order by l_returnflag, l_linestatus",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/tpch/tpch_sf0.1.sqlite")
    ap.add_argument("--out", default="lab/tpch_tasks.json")
    ap.add_argument("--check", metavar="TASKFILE")
    a = ap.parse_args()
    if a.check:
        from lab.spider2 import verify_spider
        tasks = {t["id"]: t for t in json.load(open(a.check))}
        ok = 0
        for tid, sql in SQLITE_CHECKS.items():
            r = verify_spider(tasks[tid], sql); ok += r["pass"]
            print(tid, "PASS" if r["pass"] else f"FAIL {json.dumps(r['gates'])[:200]}")
        raise SystemExit(0 if ok == len(SQLITE_CHECKS) else 1)
    tasks = build(Path(a.db))
    json.dump(tasks, open(a.out, "w"), indent=1)
    print(f"wrote {len(tasks)} tasks → {a.out} · db {a.db} {os.path.getsize(a.db)/1e6:.0f} MB · gold rows min "
          f"{min(len(t['gold_results'][0]['rows']) for t in tasks)} max {max(len(t['gold_results'][0]['rows']) for t in tasks)} · "
          f"ddl chars {len(tasks[0]['ddl'])}")


if __name__ == "__main__":
    main()
