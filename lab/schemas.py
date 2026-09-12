"""Schemas and deterministic snapshot generators for the lockbox.

Each schema has a DDL string (shown to the agent) and a `populate(con, seed)` that fills a DuckDB
connection with seeded random data. Snapshot 0 (seed 0) is the visible one the agent may query;
snapshots 1..N are hidden and used only by the verifier. Every generator injects adversarial rows:
NULLs in nullable columns, duplicate names, empty groups, exact ties, zero-quantity lines.
"""
import random
from datetime import date, timedelta

# ---------------------------------------------------------------- shop
SHOP_DDL = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name VARCHAR, country VARCHAR, signup_date DATE);
CREATE TABLE products  (id INTEGER PRIMARY KEY, name VARCHAR, category VARCHAR, unit_price DECIMAL(10,2));
CREATE TABLE orders    (id INTEGER PRIMARY KEY, customer_id INTEGER, order_date DATE, status VARCHAR);
-- status is one of 'paid', 'refunded', 'cancelled'
CREATE TABLE order_items (order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price DECIMAL(10,2));
-- unit_price on order_items is the price at time of sale and may differ from products.unit_price
"""

COUNTRIES = ["IN", "US", "DE", "BR", "JP", None]
CATEGORIES = ["books", "toys", "kitchen", "garden"]


def populate_shop(con, seed: int):
    r = random.Random(seed)
    d0 = date(2024, 1, 1)
    n_cust = r.randint(40, 60)
    customers = []
    for i in range(1, n_cust + 1):
        name = r.choice(["Asha", "Ben", "Chen", "Dev", "Eva", "Fatima", "Gus"]) + (" " + r.choice(["K", "L", "M"]) if r.random() < 0.5 else "")
        customers.append((i, name, r.choice(COUNTRIES), d0 + timedelta(days=r.randint(0, 400))))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)

    n_prod = r.randint(12, 20)
    products = []
    for i in range(1, n_prod + 1):
        products.append((i, f"p{i}", r.choice(CATEGORIES), round(r.uniform(2, 120), 2)))
    # guarantee an empty category sometimes
    if r.random() < 0.5:
        products = [(a, b, c if c != "garden" else "kitchen", d) for a, b, c, d in products]
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    n_orders = r.randint(150, 260)
    orders, items = [], []
    for oid in range(1, n_orders + 1):
        cid = r.randint(1, n_cust)
        odate = d0 + timedelta(days=r.randint(0, 500))
        status = r.choices(["paid", "refunded", "cancelled"], weights=[8, 1, 1])[0]
        orders.append((oid, cid, odate, status))
        for _ in range(r.randint(1, 4)):
            pid = r.randint(1, n_prod)
            base = products[pid - 1][3]
            price = round(base * r.choice([1.0, 1.0, 0.9, 1.1]), 2)
            qty = r.choice([0, 1, 1, 2, 3, 5]) if r.random() < 0.05 else r.randint(1, 5)
            items.append((oid, pid, qty, price))
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?)", items)


# ---------------------------------------------------------------- hr
HR_DDL = """
CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR, parent_id INTEGER);
-- parent_id references departments.id; NULL for top-level departments
CREATE TABLE employees (id INTEGER PRIMARY KEY, name VARCHAR, dept_id INTEGER, manager_id INTEGER, hire_date DATE, left_date DATE);
-- left_date is NULL for current employees; manager_id references employees.id
CREATE TABLE salaries (employee_id INTEGER, effective_from DATE, amount DECIMAL(12,2));
-- one row per salary change; the current salary is the row with the latest effective_from
CREATE TABLE reviews (employee_id INTEGER, review_year INTEGER, score INTEGER);
-- score 1..5; an employee may have no review in a given year
"""


def populate_hr(con, seed: int):
    r = random.Random(seed)
    d0 = date(2018, 1, 1)
    depts = [(1, "Corporate", None), (2, "Engineering", 1), (3, "Sales", 1), (4, "Platform", 2), (5, "Apps", 2), (6, "Field", 3)]
    if r.random() < 0.5:
        depts.append((7, "Research", 2))  # sometimes an empty department
    con.executemany("INSERT INTO departments VALUES (?,?,?)", depts)

    n_emp = r.randint(45, 70)
    emps = []
    for i in range(1, n_emp + 1):
        dept = r.choice([2, 3, 4, 5, 6])
        mgr = r.randint(1, i - 1) if i > 1 and r.random() < 0.9 else None
        hire = d0 + timedelta(days=r.randint(0, 2200))
        left = hire + timedelta(days=r.randint(30, 1500)) if r.random() < 0.2 else None
        name = r.choice(["Ira", "Jon", "Kai", "Lee", "Mo", "Nia"]) + " " + r.choice(["P", "Q", "R", "S"])
        emps.append((i, name, dept, mgr, hire, left))
    con.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", emps)

    sal, rev = [], []
    for (i, _, _, _, hire, left) in emps:
        amt = round(r.uniform(30000, 90000), -2)
        sal.append((i, hire, amt))
        for k in range(r.randint(0, 3)):
            hire2 = hire + timedelta(days=365 * (k + 1))
            if left and hire2 > left:
                break
            amt = round(amt * r.uniform(1.0, 1.15), -2)
            sal.append((i, hire2, amt))
        for y in range(2019, 2025):
            if r.random() < 0.7 and hire.year <= y and (left is None or left.year >= y):
                rev.append((i, y, r.randint(1, 5)))
    con.executemany("INSERT INTO salaries VALUES (?,?,?)", sal)
    con.executemany("INSERT INTO reviews VALUES (?,?,?)", rev)


# ---------------------------------------------------------------- events
EVENTS_DDL = """
CREATE TABLE users (id INTEGER PRIMARY KEY, plan VARCHAR, created_at TIMESTAMP);
-- plan is one of 'free', 'pro', 'team'
CREATE TABLE sessions (id INTEGER PRIMARY KEY, user_id INTEGER, started_at TIMESTAMP, ended_at TIMESTAMP, device VARCHAR);
-- ended_at is NULL for sessions that never closed cleanly
CREATE TABLE events (session_id INTEGER, ts TIMESTAMP, kind VARCHAR, value DOUBLE);
-- kind is one of 'view', 'click', 'purchase'; value is the purchase amount for 'purchase' and NULL otherwise
"""


def populate_events(con, seed: int):
    r = random.Random(seed)
    from datetime import datetime
    t0 = datetime(2025, 3, 1)
    n_users = r.randint(30, 50)
    users = [(i, r.choices(["free", "pro", "team"], weights=[6, 3, 1])[0], t0 + timedelta(hours=r.randint(0, 24 * 60))) for i in range(1, n_users + 1)]
    con.executemany("INSERT INTO users VALUES (?,?,?)", users)
    sessions, events = [], []
    sid = 0
    for (uid, plan, created) in users:
        for _ in range(r.randint(0, 8)):  # some users have zero sessions
            sid += 1
            st = created + timedelta(hours=r.randint(1, 24 * 90))
            dur = r.randint(1, 180)
            en = st + timedelta(minutes=dur) if r.random() < 0.9 else None
            sessions.append((sid, uid, st, en, r.choice(["ios", "android", "web"])))
            t = st
            for _ in range(r.randint(1, 12)):
                t = t + timedelta(seconds=r.randint(5, 600))
                kind = r.choices(["view", "click", "purchase"], weights=[7, 3, 1])[0]
                events.append((sid, t, kind, round(r.uniform(1, 200), 2) if kind == "purchase" else None))
    con.executemany("INSERT INTO sessions VALUES (?,?,?,?,?)", sessions)
    con.executemany("INSERT INTO events VALUES (?,?,?,?)", events)


SCHEMAS = {
    "shop": (SHOP_DDL, populate_shop),
    "hr": (HR_DDL, populate_hr),
    "events": (EVENTS_DDL, populate_events),
}


def build(schema: str, seed: int):
    """Return a fresh in-memory DuckDB connection populated for (schema, seed)."""
    import duckdb
    ddl, pop = SCHEMAS[schema]
    con = duckdb.connect(":memory:")
    con.execute(ddl)
    pop(con, seed)
    return con
