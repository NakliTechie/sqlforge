"""TPC-DS (scale 0.1) in SQLite as the second analytical pool: 24 tables, the 99 official queries as gold (DuckDB's
tpcds_queries() text, results from DuckDB on the same dsdgen data exported to SQLite), natural-language business
questions authored in lab/tpcds_questions.py. Same task shape as lab/tpch.py (kind spider2, SQLite engine, Spider rules).
Queries whose gold uses ROLLUP/GROUPING keep their gold (SQLite has no ROLLUP; UNION ALL reproduces it) — the 8-seed base
pass decides which tasks are learnable; questions with no author text are skipped.

Build:  uv run python -m lab.tpcds --out lab/tpcds_tasks.json --db data/tpcds/tpcds_sf0.1.sqlite
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lab.tpch import export_sqlite, norm

DOC = """TPC-DS data dictionary (scale 0.1). A retailer sells through three channels, each with a sales and a returns fact
table: store_sales/store_returns (ss_*/sr_*), catalog_sales/catalog_returns (cs_*/cr_*), web_sales/web_returns (ws_*/wr_*).
Facts reference dimensions by surrogate keys (*_sk): date_dim (d_date_sk, d_date, d_year, d_moy = month, d_dom = day,
d_qoy = quarter, d_week_seq, d_month_seq, d_day_name, d_dow), time_dim (t_time_sk, t_hour, t_minute), item (i_item_sk,
i_item_id, i_brand, i_brand_id, i_class, i_class_id, i_category, i_category_id, i_manufact_id, i_manager_id, i_current_price,
i_item_desc, i_product_name, i_color, i_size, i_units), customer (c_customer_sk, c_customer_id, c_first_name, c_last_name,
c_current_addr_sk, c_current_cdemo_sk, c_current_hdemo_sk, c_birth_year, c_birth_month, c_birth_country, c_email_address,
c_preferred_cust_flag, c_login), customer_address (ca_address_sk, ca_state, ca_county, ca_city, ca_zip, ca_country,
ca_gmt_offset), customer_demographics (cd_demo_sk, cd_gender, cd_marital_status, cd_education_status, cd_purchase_estimate,
cd_credit_rating, cd_dep_count, cd_dep_employed_count, cd_dep_college_count), household_demographics (hd_demo_sk,
hd_income_band_sk, hd_buy_potential, hd_dep_count, hd_vehicle_count), income_band (ib_income_band_sk, ib_lower_bound,
ib_upper_bound), store (s_store_sk, s_store_id, s_store_name, s_state, s_county, s_city, s_zip, s_number_employees,
s_floor_space, s_gmt_offset), warehouse (w_warehouse_sk, w_warehouse_name, w_state, w_county), ship_mode (sm_ship_mode_sk,
sm_type, sm_carrier), promotion (p_promo_sk, p_channel_email, p_channel_event, p_channel_dmail, p_channel_tv),
catalog_page (cp_catalog_page_sk, cp_catalog_page_id), web_site (web_site_sk, web_site_id, web_name), web_page
(wp_web_page_sk), reason (r_reason_sk, r_reason_desc), call_center (cc_call_center_sk, cc_call_center_id, cc_name,
cc_manager), inventory (inv_date_sk, inv_item_sk, inv_warehouse_sk, inv_quantity_on_hand). Sales facts carry
*_sold_date_sk, *_sold_time_sk, *_item_sk, *_customer_sk (store) / *_bill_customer_sk and *_ship_customer_sk (catalog,
web), *_quantity, *_list_price, *_sales_price, *_ext_sales_price, *_ext_list_price, *_ext_discount_amt, *_ext_wholesale_cost,
*_coupon_amt, *_net_paid, *_net_profit, and *_promo_sk; returns carry *_returned_date_sk, *_return_amt, *_return_quantity,
*_net_loss, *_reason_sk. Dates are TEXT 'YYYY-MM-DD'. Money columns are REAL."""


def build(db: Path) -> list[dict]:
    import duckdb
    from lab.spider2 import schema_ddl
    from lab.tpcds_questions import QUESTIONS
    con = duckdb.connect()
    con.execute("INSTALL tpcds; LOAD tpcds; CALL dsdgen(sf=0.1)")
    if not db.exists():
        export_sqlite(con, db)
    ddl = schema_ddl(str(db))
    qs = dict(con.execute("select query_nr, query from tpcds_queries()").fetchall())
    tasks, skipped = [], []
    for nr, sql in sorted(qs.items()):
        q = QUESTIONS.get(nr)
        if not q:
            skipped.append(nr); continue
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [[norm(v) for v in r] for r in cur.fetchall()]
        if not rows or all(v in (0, 0.0, None) for r in rows for v in r):
            print(f"drop q{nr}: degenerate gold ({len(rows)} rows)"); continue
        tasks.append({"kind": "spider2", "id": f"tpcds-q{nr:02d}", "schema": "tpcds", "db_path": str(db), "hops": 5,
                      "q": q + "\n\nReference document (tpcds data dictionary):\n" + DOC, "ddl": ddl, "gold": [],
                      "gold_sql": None, "gold_sql_duckdb": sql, "gold_results": [{"file": None, "cols": cols, "rows": rows}],
                      "condition_cols": [], "ignore_order": "order by" not in sql.lower(), "external_knowledge": "tpcds"})
    if skipped:
        print(f"no question yet for {len(skipped)} queries: {skipped}")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/tpcds/tpcds_sf0.1.sqlite")
    ap.add_argument("--out", default="lab/tpcds_tasks.json")
    a = ap.parse_args()
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    tasks = build(Path(a.db))
    json.dump(tasks, open(a.out, "w"), indent=1)
    print(f"wrote {len(tasks)} tasks → {a.out} · db {a.db} {os.path.getsize(a.db)/1e6:.0f} MB · ddl chars {len(tasks[0]['ddl']) if tasks else 0}")


if __name__ == "__main__":
    main()
