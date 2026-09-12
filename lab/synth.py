"""Compositional task synthesis — gold SQL correct by construction. Lockbox.

Tasks are built from hop primitives per schema. Each primitive contributes a CTE (or the final
select) plus an English clause; composing k primitives yields a k-hop question with its gold.
Parameters (filters, aggregates, thresholds, tie-breaks) are drawn from a seeded RNG, so the
space is large and reproducible. Every candidate then passes the same admission checks as the
hand-made set: gold executes on all hidden seeds, non-empty on the visible seed, result differs
across seeds (no constant answers), and deduped by normalised gold SQL.

Usage: uv run python -m lab.synth --n 200 --seed 7 --out lab/synth_tasks.json
"""
from __future__ import annotations

import argparse
import json
import random

import sqlglot

from lab import schemas, verify
from lab.tasks import DDL, HIDDEN_SEEDS, VISIBLE_SEED

# ----------------------------------------------------------------------------- shop
STATUSES = ["paid", "refunded", "cancelled"]


def shop_tasks(r: random.Random) -> list[dict]:
    st = r.choice(STATUSES)
    out = []
    # 1 hop
    out.append(dict(hops=1, ignore_order=False,
        q=f"How many orders have status '{st}'? Return a single number.",
        gold=f"SELECT COUNT(*) AS n FROM orders WHERE status = '{st}'"))
    agg = r.choice([("SUM", "total revenue (sum of quantity * order_items.unit_price)"), ("AVG", "average line value (quantity * order_items.unit_price)"), ("COUNT", "number of order lines")])
    expr = "oi.quantity * oi.unit_price" if agg[0] != "COUNT" else "*"
    grp = r.choice([("p.category", "product category", "category"), ("c.country", "customer country (NULL countries excluded)", "country")])
    extra = " AND c.country IS NOT NULL" if grp[2] == "country" else ""
    out.append(dict(hops=1, ignore_order=True,
        q=f"For orders with status '{st}' only, what is the {agg[1]} per {grp[1]}? Return {grp[2]} and the value.",
        gold=f"SELECT {grp[0]}, {agg[0]}({expr}) AS v FROM order_items oi JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id JOIN customers c ON c.id = o.customer_id WHERE o.status = '{st}'{extra} GROUP BY {grp[0]}"))
    # 2 hops: group → top-1 with tie-break
    k = r.choice([1, 2, 3])
    out.append(dict(hops=2, ignore_order=False,
        q=f"Which {k} customer{'s' if k>1 else ''} have the highest total revenue from '{st}' orders (sum of quantity * order_items.unit_price)? Return customer id and total, highest first; on ties the lower id first.",
        gold=f"SELECT c.id, SUM(oi.quantity * oi.unit_price) AS total FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id WHERE o.status = '{st}' GROUP BY c.id ORDER BY total DESC, c.id LIMIT {k}"))
    qty = r.choice([0, 1, 5])
    out.append(dict(hops=2, ignore_order=False,
        q=f"How many '{st}' orders contain at least one order line with quantity {qty}? Return a single number.",
        gold=f"SELECT COUNT(*) AS n FROM orders o WHERE o.status = '{st}' AND EXISTS (SELECT 1 FROM order_items oi WHERE oi.order_id = o.id AND oi.quantity = {qty})"))
    # 3 hops: per-customer first order → year → count ; HAVING threshold
    fn = r.choice([("MIN", "first"), ("MAX", "latest")])
    out.append(dict(hops=3, ignore_order=True,
        q=f"For each customer, find the date of their {fn[1]} '{st}' order. Then count customers by the calendar year of that date. Return year and count.",
        gold=f"WITH f AS (SELECT customer_id, {fn[0]}(order_date) AS d FROM orders WHERE status = '{st}' GROUP BY customer_id) SELECT EXTRACT(year FROM d) AS yr, COUNT(*) AS n FROM f GROUP BY yr"))
    th = r.choice([5, 10, 15])
    out.append(dict(hops=3, ignore_order=True,
        q=f"What is the average '{st}' order value (an order's value is the sum of quantity * order_items.unit_price over its lines) per customer country, only for countries with at least {th} such orders and excluding NULL countries? Return country and average value.",
        gold=f"WITH ov AS (SELECT o.id, c.country, SUM(oi.quantity * oi.unit_price) AS v FROM orders o JOIN customers c ON c.id = o.customer_id JOIN order_items oi ON oi.order_id = o.id WHERE o.status = '{st}' AND c.country IS NOT NULL GROUP BY o.id, c.country) SELECT country, AVG(v) AS avg_value FROM ov GROUP BY country HAVING COUNT(*) >= {th}"))
    out.append(dict(hops=3, ignore_order=True,
        q=f"For each product category, which product earned the most revenue from '{st}' orders (sum of quantity * order_items.unit_price)? Return category and product name; on ties within a category pick the lowest product id.",
        gold=f"WITH r AS (SELECT p.category, p.id, p.name, SUM(oi.quantity * oi.unit_price) AS rev FROM products p JOIN order_items oi ON oi.product_id = p.id JOIN orders o ON o.id = oi.order_id WHERE o.status = '{st}' GROUP BY p.category, p.id, p.name), k AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY rev DESC, id) AS rn FROM r) SELECT category, name FROM k WHERE rn = 1"))
    # 4 hops: window first/second → gap → count ; repeat-rate percent
    days = r.choice([14, 30, 60, 90])
    nth = r.choice([2, 3])
    out.append(dict(hops=4, ignore_order=False,
        q=f"How many customers placed their {'second' if nth==2 else 'third'} '{st}' order within {days} days (inclusive) of their first '{st}' order? Order by order_date, then order id, to rank orders. Return a single number.",
        gold=f"WITH s AS (SELECT customer_id, order_date, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date, id) AS rn FROM orders WHERE status = '{st}'), f AS (SELECT customer_id, order_date AS d1 FROM s WHERE rn = 1), g AS (SELECT customer_id, order_date AS d2 FROM s WHERE rn = {nth}) SELECT COUNT(*) AS n FROM f JOIN g USING (customer_id) WHERE d2 - d1 <= {days}"))
    m = r.choice([2, 3, 4])
    out.append(dict(hops=4, ignore_order=False,
        q=f"Among customers with at least one '{st}' order, what percentage have {m} or more '{st}' orders? Return a single number rounded to 2 decimal places.",
        gold=f"WITH c AS (SELECT customer_id, COUNT(*) AS n FROM orders WHERE status = '{st}' GROUP BY customer_id) SELECT ROUND(100.0 * SUM(CASE WHEN n >= {m} THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct FROM c"))
    return [dict(schema="shop", **t) for t in out]


# ----------------------------------------------------------------------------- hr
def hr_tasks(r: random.Random) -> list[dict]:
    out = []
    cur = r.choice([("left_date IS NULL", "current"), ("left_date IS NOT NULL", "former")])
    out.append(dict(hops=1, ignore_order=False,
        q=f"How many employees are {cur[1]} ({'left_date is NULL' if cur[1]=='current' else 'left_date is not NULL'})? Return a single number.",
        gold=f"SELECT COUNT(*) AS n FROM employees WHERE {cur[0]}"))
    out.append(dict(hops=1, ignore_order=True,
        q=f"How many {cur[1]} employees are in each department? Include departments with zero. Return department name and count.",
        gold=f"SELECT d.name, COUNT(e.id) AS n FROM departments d LEFT JOIN employees e ON e.dept_id = d.id AND e.{cur[0]} GROUP BY d.name"))
    agg = r.choice([("AVG", "average"), ("MAX", "highest"), ("MIN", "lowest")])
    out.append(dict(hops=2, ignore_order=False,
        q=f"What is the {agg[1]} current salary among {cur[1]} employees? Current salary = the salary row with the latest effective_from. Return a single number rounded to 2 decimal places.",
        gold=f"WITH cur AS (SELECT employee_id, amount, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY effective_from DESC) AS rn FROM salaries) SELECT ROUND({agg[0]}(cur.amount), 2) AS v FROM cur JOIN employees e ON e.id = cur.employee_id WHERE cur.rn = 1 AND e.{cur[0]}"))
    out.append(dict(hops=2, ignore_order=False,
        q=f"Which department has the highest average current salary among its {cur[1]} employees (current salary = latest effective_from)? Return the department name; on ties the alphabetically first.",
        gold=f"WITH cur AS (SELECT employee_id, amount, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY effective_from DESC) AS rn FROM salaries) SELECT d.name FROM cur JOIN employees e ON e.id = cur.employee_id JOIN departments d ON d.id = e.dept_id WHERE cur.rn = 1 AND e.{cur[0]} GROUP BY d.name ORDER BY AVG(cur.amount) DESC, d.name LIMIT 1"))
    yr = r.choice([2020, 2021, 2022, 2023])
    th = r.choice([2, 3, 5])
    out.append(dict(hops=3, ignore_order=True,
        q=f"What is the average review score per department for review_year {yr}, counting only departments with at least {th} reviews that year? Return department name and average score.",
        gold=f"SELECT d.name, AVG(r.score) AS avg_score FROM reviews r JOIN employees e ON e.id = r.employee_id JOIN departments d ON d.id = e.dept_id WHERE r.review_year = {yr} GROUP BY d.name HAVING COUNT(*) >= {th}"))
    k = r.choice([1, 3, 5])
    out.append(dict(hops=3, ignore_order=False,
        q=f"For each current manager, count their current direct reports. Return the top {k} managers as (manager name, count), ordered by count descending, then manager id ascending.",
        gold=f"SELECT m.name, COUNT(*) AS n FROM employees e JOIN employees m ON m.id = e.manager_id WHERE e.left_date IS NULL AND m.left_date IS NULL GROUP BY m.id, m.name ORDER BY n DESC, m.id LIMIT {k}"))
    out.append(dict(hops=3, ignore_order=False,
        q=f"Which {cur[1]} employee has the largest salary increase (current salary with the latest effective_from minus the first salary with the earliest effective_from)? Return employee id and the increase; on ties the lowest id.",
        gold=f"WITH s AS (SELECT employee_id, amount, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY effective_from DESC) AS rd, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY effective_from ASC) AS ra FROM salaries), c AS (SELECT employee_id, amount FROM s WHERE rd = 1), f AS (SELECT employee_id, amount FROM s WHERE ra = 1) SELECT e.id, c.amount - f.amount AS increase FROM employees e JOIN c ON c.employee_id = e.id JOIN f ON f.employee_id = e.id WHERE e.{cur[0]} ORDER BY increase DESC, e.id LIMIT 1"))
    out.append(dict(hops=4, ignore_order=True,
        q=f"Departments form a tree via parent_id with 'Corporate' at the root. For each department directly under Corporate, count the {cur[1]} employees in it and all its descendant departments. Return department name and headcount.",
        gold=f"WITH RECURSIVE t AS (SELECT id, name AS top_name FROM departments WHERE parent_id = (SELECT id FROM departments WHERE name = 'Corporate') UNION ALL SELECT d.id, t.top_name FROM departments d JOIN t ON d.parent_id = t.id) SELECT t.top_name AS name, COUNT(e.id) AS headcount FROM t LEFT JOIN employees e ON e.dept_id = t.id AND e.{cur[0]} GROUP BY t.top_name"))
    cut = r.choice(["2021-01-01", "2022-01-01", "2023-01-01"])
    dd = r.choice([180, 365, 730])
    out.append(dict(hops=4, ignore_order=False,
        q=f"Among employees hired before {cut}, what percentage left within {dd} days of their hire date (left_date - hire_date <= {dd})? Return a single number rounded to 2 decimal places.",
        gold=f"SELECT ROUND(100.0 * SUM(CASE WHEN left_date IS NOT NULL AND left_date - hire_date <= {dd} THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct FROM employees WHERE hire_date < DATE '{cut}'"))
    return [dict(schema="hr", **t) for t in out]


# ----------------------------------------------------------------------------- events
def events_tasks(r: random.Random) -> list[dict]:
    out = []
    kind = r.choice(["view", "click", "purchase"])
    out.append(dict(hops=1, ignore_order=True,
        q=f"How many '{kind}' events are there per session device? Return device and count.",
        gold=f"SELECT s.device, COUNT(*) AS n FROM events e JOIN sessions s ON s.id = e.session_id WHERE e.kind = '{kind}' GROUP BY s.device"))
    plan = r.choice(["free", "pro", "team"])
    out.append(dict(hops=1, ignore_order=False,
        q=f"What is the total purchase amount (sum of value over 'purchase' events) for users on the '{plan}' plan? Return a single number.",
        gold=f"SELECT COALESCE(SUM(e.value), 0) AS total FROM events e JOIN sessions s ON s.id = e.session_id JOIN users u ON u.id = s.user_id WHERE e.kind = 'purchase' AND u.plan = '{plan}'"))
    dev = r.choice(["ios", "android", "web"])
    out.append(dict(hops=2, ignore_order=False,
        q=f"What is the average duration in minutes of '{dev}' sessions that have an ended_at? Return a single number rounded to 2 decimal places.",
        gold=f"SELECT ROUND(AVG(EXTRACT(epoch FROM (ended_at - started_at)) / 60.0), 2) AS m FROM sessions WHERE ended_at IS NOT NULL AND device = '{dev}'"))
    out.append(dict(hops=2, ignore_order=False,
        q=f"How many users on the '{plan}' plan have no sessions at all? Return a single number.",
        gold=f"SELECT COUNT(*) AS n FROM users u WHERE u.plan = '{plan}' AND NOT EXISTS (SELECT 1 FROM sessions s WHERE s.user_id = u.id)"))
    out.append(dict(hops=3, ignore_order=True,
        q=f"For each user with at least one session, take the device of their earliest session (earliest started_at, then lowest session id). Count '{plan}' users by that first device. Return device and count.",
        gold=f"WITH f AS (SELECT s.user_id, s.device, ROW_NUMBER() OVER (PARTITION BY s.user_id ORDER BY s.started_at, s.id) AS rn FROM sessions s) SELECT f.device, COUNT(*) AS n FROM f JOIN users u ON u.id = f.user_id WHERE f.rn = 1 AND u.plan = '{plan}' GROUP BY f.device"))
    out.append(dict(hops=3, ignore_order=False,
        q=f"What percentage of '{dev}' sessions contain at least one '{kind}' event? Return a single number rounded to 2 decimal places.",
        gold=f"SELECT ROUND(100.0 * SUM(CASE WHEN EXISTS (SELECT 1 FROM events e WHERE e.session_id = s.id AND e.kind = '{kind}') THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct FROM sessions s WHERE s.device = '{dev}'"))
    stat = r.choice([("MEDIAN", "median"), ("MAX", "maximum"), ("AVG", "average")])
    out.append(dict(hops=3, ignore_order=False,
        q=f"What is the {stat[1]} number of events per session, over sessions that have at least one event? Return a single number.",
        gold=f"WITH c AS (SELECT session_id, COUNT(*) AS n FROM events GROUP BY session_id) SELECT {stat[0]}(n) AS v FROM c"))
    out.append(dict(hops=4, ignore_order=False,
        q=f"How many '{plan}' users made their first '{kind}' event in a session that was not their first session? Order sessions by started_at then session id; a user's first '{kind}' event is their earliest such event by ts. Return a single number.",
        gold=f"WITH s AS (SELECT id, user_id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY started_at, id) AS rn FROM sessions), p AS (SELECT s.user_id, s.rn, ROW_NUMBER() OVER (PARTITION BY s.user_id ORDER BY e.ts) AS prn FROM events e JOIN s ON s.id = e.session_id WHERE e.kind = '{kind}') SELECT COUNT(*) AS n FROM p JOIN users u ON u.id = p.user_id WHERE p.prn = 1 AND p.rn > 1 AND u.plan = '{plan}'"))
    unit = r.choice(["week", "month"])
    out.append(dict(hops=4, ignore_order=False,
        q=f"Bucket sessions by the {unit} they started (DATE_TRUNC('{unit}', started_at)). Which {unit} had the most distinct active users? Return the {unit} start as a date and the user count; on ties the earliest {unit}.",
        gold=f"SELECT CAST(DATE_TRUNC('{unit}', started_at) AS DATE) AS b, COUNT(DISTINCT user_id) AS n FROM sessions GROUP BY b ORDER BY n DESC, b LIMIT 1"))
    return [dict(schema="events", **t) for t in out]


GENERATORS = [shop_tasks, hr_tasks, events_tasks]


def _norm_sql(sql: str) -> str:
    return sqlglot.transpile(sql, read="duckdb", write="duckdb", pretty=False)[0].lower()


def admit(task: dict) -> tuple[bool, str]:
    """Cheap admission gates (no model): executes on hidden seeds, non-empty on visible, varies across seeds."""
    res = verify.verify(task, task["gold"][0], HIDDEN_SEEDS)
    if not res["pass"]:
        return False, f"gold fails ladder at {verify.first_failed_gate(res)}"
    outs = []
    for seed in [VISIBLE_SEED] + HIDDEN_SEEDS:
        con = schemas.build(task["schema"], seed)
        cols, rows = verify.execute(con, task["gold"][0])
        if seed == VISIBLE_SEED and not rows:
            return False, "empty on visible"
        outs.append(tuple(sorted(map(str, rows))))
    if len(set(outs)) == 1:
        return False, "constant answer across seeds"
    return True, ""


def generate(n: int, seed: int) -> tuple[list[dict], dict]:
    r = random.Random(seed)
    seen, tasks, rejected = set(), [], {}
    i = 0
    while len(tasks) < n and i < n * 20:
        i += 1
        gen = r.choice(GENERATORS)
        for t in gen(random.Random(r.random())):
            t["ddl"] = DDL[t["schema"]]
            t["gold"] = [t.pop("gold")]
            t["condition_cols"] = None
            key = _norm_sql(t["gold"][0])
            if key in seen:
                continue
            ok, why = admit(t)
            if not ok:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            seen.add(key)
            t["id"] = f"syn{len(tasks):04d}"
            tasks.append(t)
            if len(tasks) >= n:
                break
    return tasks, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="lab/synth_tasks.json")
    a = ap.parse_args()
    tasks, rejected = generate(a.n, a.seed)
    json.dump(tasks, open(a.out, "w"), indent=1, default=str)
    from collections import Counter
    print(f"admitted {len(tasks)} unique tasks → {a.out}")
    print("by hops", dict(sorted(Counter(t["hops"] for t in tasks).items())))
    print("by schema", dict(Counter(t["schema"] for t in tasks)))
    print("rejected", rejected)


if __name__ == "__main__":
    main()
