"""The fixed 30-task set for the harness-fit campaign. Lockbox.

hops = number of dependent steps a solver must chain (filter/aggregate → use that result → ...).
Each gold is DuckDB SQL; ignore_order=True where the question does not fix an order;
condition_cols (0-based gold column indexes) where only some output columns are graded.
"""
from lab.schemas import SHOP_DDL, HR_DDL, EVENTS_DDL

TASKS = [
    # ------------------------------------------------------------ shop
    dict(id="shop01", schema="shop", hops=1, ignore_order=False,
         q="How many orders have status 'paid'? Return a single number.",
         gold=["SELECT COUNT(*) AS n FROM orders WHERE status = 'paid'"]),
    dict(id="shop02", schema="shop", hops=1, ignore_order=True,
         q="For paid orders only, what is the total revenue (sum of quantity * order_items.unit_price) per product category? Return category and revenue.",
         gold=["SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id WHERE o.status = 'paid' GROUP BY p.category"]),
    dict(id="shop03", schema="shop", hops=2, ignore_order=False,
         q="Which customer has the highest total revenue from paid orders (sum of quantity * order_items.unit_price)? Return customer id and that total. On ties return the lowest customer id.",
         gold=["SELECT c.id, SUM(oi.quantity * oi.unit_price) AS total FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id WHERE o.status = 'paid' GROUP BY c.id ORDER BY total DESC, c.id LIMIT 1"]),
    dict(id="shop04", schema="shop", hops=2, ignore_order=False,
         q="Which country (ignore customers whose country is NULL) has the most distinct customers with at least one paid order? Return the country code and the count. On ties return the alphabetically first country.",
         gold=["SELECT c.country, COUNT(DISTINCT c.id) AS n FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.status = 'paid' AND c.country IS NOT NULL GROUP BY c.country ORDER BY n DESC, c.country LIMIT 1"]),
    dict(id="shop05", schema="shop", hops=2, ignore_order=False,
         q="How many paid orders contain at least one order line with quantity 0? Return a single number.",
         gold=["SELECT COUNT(*) AS n FROM orders o WHERE o.status = 'paid' AND EXISTS (SELECT 1 FROM order_items oi WHERE oi.order_id = o.id AND oi.quantity = 0)"]),
    dict(id="shop06", schema="shop", hops=3, ignore_order=True,
         q="For each customer, find the date of their first paid order. Then count how many customers had their first paid order in each calendar year. Return year and count.",
         gold=["WITH f AS (SELECT customer_id, MIN(order_date) AS first_dt FROM orders WHERE status = 'paid' GROUP BY customer_id) SELECT EXTRACT(year FROM first_dt) AS yr, COUNT(*) AS n FROM f GROUP BY yr"]),
    dict(id="shop07", schema="shop", hops=3, ignore_order=True,
         q="What is the average paid order value (an order's value is the sum of quantity * order_items.unit_price over its lines) per customer country, considering only countries with at least 10 paid orders and excluding NULL countries? Return country and average value.",
         gold=["WITH ov AS (SELECT o.id, c.country, SUM(oi.quantity * oi.unit_price) AS v FROM orders o JOIN customers c ON c.id = o.customer_id JOIN order_items oi ON oi.order_id = o.id WHERE o.status = 'paid' AND c.country IS NOT NULL GROUP BY o.id, c.country) SELECT country, AVG(v) AS avg_value FROM ov GROUP BY country HAVING COUNT(*) >= 10"]),
    dict(id="shop08", schema="shop", hops=3, ignore_order=True,
         q="For each product category, which product earned the most revenue from paid orders (sum of quantity * order_items.unit_price)? Return category and product name. On ties within a category pick the lowest product id.",
         gold=["WITH r AS (SELECT p.category, p.id, p.name, SUM(oi.quantity * oi.unit_price) AS rev FROM products p JOIN order_items oi ON oi.product_id = p.id JOIN orders o ON o.id = oi.order_id WHERE o.status = 'paid' GROUP BY p.category, p.id, p.name), k AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY rev DESC, id) AS rn FROM r) SELECT category, name FROM k WHERE rn = 1"]),
    dict(id="shop09", schema="shop", hops=4, ignore_order=False,
         q="How many customers placed their second paid order within 30 days (inclusive) of their first paid order? Order by order_date, then order id, to decide first and second. Return a single number.",
         gold=["WITH s AS (SELECT customer_id, order_date, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date, id) AS rn FROM orders WHERE status = 'paid'), f AS (SELECT customer_id, order_date AS d1 FROM s WHERE rn = 1), g AS (SELECT customer_id, order_date AS d2 FROM s WHERE rn = 2) SELECT COUNT(*) AS n FROM f JOIN g USING (customer_id) WHERE d2 - d1 <= 30"]),
    dict(id="shop10", schema="shop", hops=4, ignore_order=False,
         q="Among customers with at least one paid order, what percentage have two or more paid orders? Return a single number rounded to 2 decimal places (e.g. 37.50).",
         gold=["WITH c AS (SELECT customer_id, COUNT(*) AS n FROM orders WHERE status = 'paid' GROUP BY customer_id) SELECT ROUND(100.0 * SUM(CASE WHEN n >= 2 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct FROM c"]),
    # ------------------------------------------------------------ hr
    dict(id="hr01", schema="hr", hops=1, ignore_order=False,
         q="How many employees are current (left_date is NULL)? Return a single number.",
         gold=["SELECT COUNT(*) AS n FROM employees WHERE left_date IS NULL"]),
    dict(id="hr02", schema="hr", hops=1, ignore_order=True,
         q="How many employees (current or former) are in each department? Include departments with zero employees. Return department name and count.",
         gold=["SELECT d.name, COUNT(e.id) AS n FROM departments d LEFT JOIN employees e ON e.dept_id = d.id GROUP BY d.name"]),
    dict(id="hr03", schema="hr", hops=2, ignore_order=False,
         q="What is the average current salary of current employees? An employee's current salary is the amount on their salary row with the latest effective_from. Return a single number rounded to 2 decimal places.",
         gold=["WITH cur AS (SELECT employee_id, amount, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY effective_from DESC) AS rn FROM salaries) SELECT ROUND(AVG(cur.amount), 2) AS avg_salary FROM cur JOIN employees e ON e.id = cur.employee_id WHERE cur.rn = 1 AND e.left_date IS NULL"]),
    dict(id="hr04", schema="hr", hops=2, ignore_order=False,
         q="Which department has the highest average current salary among its current employees? Current salary = the row with the latest effective_from. Return the department name. On ties return the alphabetically first name.",
         gold=["WITH cur AS (SELECT employee_id, amount, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY effective_from DESC) AS rn FROM salaries) SELECT d.name FROM cur JOIN employees e ON e.id = cur.employee_id JOIN departments d ON d.id = e.dept_id WHERE cur.rn = 1 AND e.left_date IS NULL GROUP BY d.name ORDER BY AVG(cur.amount) DESC, d.name LIMIT 1"]),
    dict(id="hr05", schema="hr", hops=2, ignore_order=False,
         q="How many employees have never had a review? Return a single number.",
         gold=["SELECT COUNT(*) AS n FROM employees e WHERE NOT EXISTS (SELECT 1 FROM reviews r WHERE r.employee_id = e.id)"]),
    dict(id="hr06", schema="hr", hops=3, ignore_order=False,
         q="For each current manager, count their current direct reports. Return the top 3 managers by that count as (manager name, count), ordered by count descending, then manager id ascending.",
         gold=["SELECT m.name, COUNT(*) AS n FROM employees e JOIN employees m ON m.id = e.manager_id WHERE e.left_date IS NULL AND m.left_date IS NULL GROUP BY m.id, m.name ORDER BY n DESC, m.id LIMIT 3"]),
    dict(id="hr07", schema="hr", hops=3, ignore_order=True,
         q="What is the average review score per department for review_year 2023, counting only departments with at least 3 reviews that year? Return department name and average score.",
         gold=["SELECT d.name, AVG(r.score) AS avg_score FROM reviews r JOIN employees e ON e.id = r.employee_id JOIN departments d ON d.id = e.dept_id WHERE r.review_year = 2023 GROUP BY d.name HAVING COUNT(*) >= 3"]),
    dict(id="hr08", schema="hr", hops=3, ignore_order=False,
         q="Which current employee has the largest salary increase, defined as current salary (latest effective_from) minus first salary (earliest effective_from)? Return employee id and the increase. On ties return the lowest id.",
         gold=["WITH s AS (SELECT employee_id, amount, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY effective_from DESC) AS rd, ROW_NUMBER() OVER (PARTITION BY employee_id ORDER BY effective_from ASC) AS ra FROM salaries), cur AS (SELECT employee_id, amount FROM s WHERE rd = 1), fst AS (SELECT employee_id, amount FROM s WHERE ra = 1) SELECT e.id, cur.amount - fst.amount AS increase FROM employees e JOIN cur ON cur.employee_id = e.id JOIN fst ON fst.employee_id = e.id WHERE e.left_date IS NULL ORDER BY increase DESC, e.id LIMIT 1"]),
    dict(id="hr09", schema="hr", hops=4, ignore_order=True,
         q="Departments form a tree via parent_id with 'Corporate' at the root. For each department directly under Corporate, count the current employees in it and in all its descendant departments. Return department name and headcount.",
         gold=["WITH RECURSIVE t AS (SELECT id, name AS top_name, id AS top_id FROM departments WHERE parent_id = (SELECT id FROM departments WHERE name = 'Corporate') UNION ALL SELECT d.id, t.top_name, t.top_id FROM departments d JOIN t ON d.parent_id = t.id) SELECT t.top_name AS name, COUNT(e.id) AS headcount FROM t LEFT JOIN employees e ON e.dept_id = t.id AND e.left_date IS NULL GROUP BY t.top_name"]),
    dict(id="hr10", schema="hr", hops=4, ignore_order=False,
         q="Among employees hired before 2023-01-01, what percentage left within one year of their hire date (left_date - hire_date <= 365 days)? Return a single number rounded to 2 decimal places.",
         gold=["SELECT ROUND(100.0 * SUM(CASE WHEN left_date IS NOT NULL AND left_date - hire_date <= 365 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct FROM employees WHERE hire_date < DATE '2023-01-01'"]),
    # ------------------------------------------------------------ events
    dict(id="ev01", schema="events", hops=1, ignore_order=True,
         q="How many sessions are there per device? Return device and count.",
         gold=["SELECT device, COUNT(*) AS n FROM sessions GROUP BY device"]),
    dict(id="ev02", schema="events", hops=1, ignore_order=False,
         q="What is the total purchase amount across all 'purchase' events? Return a single number.",
         gold=["SELECT SUM(value) AS total FROM events WHERE kind = 'purchase'"]),
    dict(id="ev03", schema="events", hops=2, ignore_order=False,
         q="What is the average session duration in minutes, over sessions that have an ended_at? Return a single number rounded to 2 decimal places.",
         gold=["SELECT ROUND(AVG(EXTRACT(epoch FROM (ended_at - started_at)) / 60.0), 2) AS avg_minutes FROM sessions WHERE ended_at IS NOT NULL"]),
    dict(id="ev04", schema="events", hops=2, ignore_order=False,
         q="Which plan has the highest purchase amount per user, defined as total purchase value from that plan's users divided by the number of users on the plan (including users with no purchases)? Return the plan. On ties return the alphabetically first plan.",
         gold=["WITH pu AS (SELECT u.plan, u.id, COALESCE(SUM(e.value), 0) AS v FROM users u LEFT JOIN sessions s ON s.user_id = u.id LEFT JOIN events e ON e.session_id = s.id AND e.kind = 'purchase' GROUP BY u.plan, u.id) SELECT plan FROM pu GROUP BY plan ORDER BY SUM(v) / COUNT(*) DESC, plan LIMIT 1"]),
    dict(id="ev05", schema="events", hops=2, ignore_order=False,
         q="How many users have no sessions at all? Return a single number.",
         gold=["SELECT COUNT(*) AS n FROM users u WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.user_id = u.id)"]),
    dict(id="ev06", schema="events", hops=3, ignore_order=True,
         q="For each user with at least one session, take the device of their earliest session (earliest started_at, then lowest session id). Count users by that first device. Return device and count.",
         gold=["WITH f AS (SELECT user_id, device, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY started_at, id) AS rn FROM sessions) SELECT device, COUNT(*) AS n FROM f WHERE rn = 1 GROUP BY device"]),
    dict(id="ev07", schema="events", hops=3, ignore_order=False,
         q="What percentage of sessions contain at least one 'purchase' event? Return a single number rounded to 2 decimal places.",
         gold=["SELECT ROUND(100.0 * SUM(CASE WHEN EXISTS (SELECT 1 FROM events e WHERE e.session_id = s.id AND e.kind = 'purchase') THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct FROM sessions s"]),
    dict(id="ev08", schema="events", hops=3, ignore_order=False,
         q="What is the median number of events per session, over sessions that have at least one event? Return a single number.",
         gold=["WITH c AS (SELECT session_id, COUNT(*) AS n FROM events GROUP BY session_id) SELECT MEDIAN(n) AS med FROM c"]),
    dict(id="ev09", schema="events", hops=4, ignore_order=False,
         q="How many users made their first purchase in a session that was not their first session? Order sessions by started_at then session id; a user's first purchase is their earliest 'purchase' event by ts. Return a single number.",
         gold=["WITH s AS (SELECT id, user_id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY started_at, id) AS rn FROM sessions), p AS (SELECT s.user_id, s.rn, ROW_NUMBER() OVER (PARTITION BY s.user_id ORDER BY e.ts) AS prn FROM events e JOIN s ON s.id = e.session_id WHERE e.kind = 'purchase') SELECT COUNT(*) AS n FROM p WHERE prn = 1 AND rn > 1"]),
    dict(id="ev10", schema="events", hops=4, ignore_order=False,
         q="Bucket sessions by the week they started (DATE_TRUNC('week', started_at)). Which week had the most distinct active users? Return the week start (as a date) and the user count. On ties return the earliest week.",
         gold=["SELECT CAST(DATE_TRUNC('week', started_at) AS DATE) AS wk, COUNT(DISTINCT user_id) AS n FROM sessions GROUP BY wk ORDER BY n DESC, wk LIMIT 1"]),
]

DDL = {"shop": SHOP_DDL, "hr": HR_DDL, "events": EVENTS_DDL}

for t in TASKS:
    t["ddl"] = DDL[t["schema"]]
    t.setdefault("condition_cols", None)

VISIBLE_SEED = 0
HIDDEN_SEEDS = [101, 202, 303]


def by_id(tid: str) -> dict:
    return next(t for t in TASKS if t["id"] == tid)
