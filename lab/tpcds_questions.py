"""Natural-language business questions for the TPC-DS queries (DuckDB tpcds_queries() instantiation, sf 0.1), authored
from the SQL so that the query's result is the answer: every filter, output column, grouping, ordering and row limit is
stated. ROLLUP queries ask for subtotals and a grand total in ROLLUP order (rolled-up columns NULL, sorted NULLS FIRST).
Queries with no entry (e.g. Q8's ~400-zip list) are skipped by lab/tpcds.py. Authored 2026-09-26 by Claude for sqlforge."""

QUESTIONS = {
 1: "Customers in Tennessee stores who returned more than 20 % above the average: using store returns dated in the year 2000, "
    "compute each customer's total return amount per store. List the customer ids (c_customer_id) of customers whose total "
    "for a store exceeds 1.2 times the average per-customer total for that same store, for stores in state 'TN'. Order by "
    "customer id; first 100.",
 2: "Week-over-week ratios of combined web and catalog sales by weekday: for every week (d_week_seq), sum the extended sales "
    "price of web sales plus catalog sales for each day name Sunday..Saturday. Pair each week of 2001 with the week of 2002 "
    "whose d_week_seq is exactly 53 larger. Return the 2001 week's d_week_seq and, for Sunday through Saturday in order, the "
    "2001 sum divided by the 2002 sum rounded to 2 decimals (8 columns). Order by the 2001 week sequence, NULLs first.",
 3: "For store sales of items from manufacturer id 128 sold in November (month 11) of any year: total extended sales price "
    "by year, brand id and brand. Return d_year, i_brand_id, i_brand and the sum, ordered by year, then sum descending, then "
    "brand id; first 100.",
 4: "Customers whose catalog-sales growth from 2001 to 2002 beat both their store-sales growth and their web-sales growth. "
    "Per customer, channel and year, yearly total = sum of ((ext_list_price - ext_wholesale_cost - ext_discount_amt) + "
    "ext_sales_price) / 2 over that channel's sales (store: ss_customer_sk; catalog: cs_bill_customer_sk; web: "
    "ws_bill_customer_sk). Keep customers with a positive 2001 total in all three channels and where the catalog ratio "
    "(2002 total / 2001 total) is greater than both the store ratio and the web ratio. Return c_customer_id, first name, "
    "last name and preferred-customer flag, ordered by those four columns (NULLs first); first 100.",
 5: "Sales, returns and profit by channel for 2000-08-23 to 2000-09-06 inclusive (by sold date for sales, returned date for "
    "returns): store channel per store (id = 'store' || s_store_id): sales = sum of ss_ext_sales_price, returns = sum of "
    "sr_return_amt, profit = sum of ss_net_profit minus sum of sr_net_loss; catalog channel per catalog page (id = "
    "'catalog_page' || cp_catalog_page_id) with cs_ext_sales_price / cr_return_amount / cs_net_profit minus cr_net_loss; web "
    "channel per web site (id = 'web_site' || web_site_id) with ws_ext_sales_price / wr_return_amt / ws_net_profit minus "
    "wr_net_loss, where a web return is attributed to the site of its matching web sale (join on item and order number). "
    "Return channel, id, sales, returns, profit with ROLLUP subtotals per channel and a grand total (rolled-up columns "
    "NULL), ordered by channel then id, NULLs first; first 100.",
 6: "States with at least 10 store-sales of overpriced items in January 2001: count store sales in the month sequence "
    "(d_month_seq) of 2001-01 where the item's current price exceeds 1.2 times the average current price of items in its "
    "category, grouped by the customer's current address state. Return state and count for counts >= 10, ordered by count "
    "then state (NULLs first); first 100.",
 7: "For store sales in the year 2000 to male, single, college-educated customers (customer_demographics: gender 'M', "
    "marital status 'S', education 'College') under promotions with no email channel or no event channel (p_channel_email "
    "= 'N' or p_channel_event = 'N'): per item id, the average quantity, average list price, average coupon amount and "
    "average sales price. Order by item id; first 100.",
 9: "Five quantity buckets of store sales (ss_quantity 1-20, 21-40, 41-60, 61-80, 81-100): for each bucket report a single "
    "value: the average ext_discount_amt when the bucket's row count exceeds its threshold (74129, 122840, 56580, 10097, "
    "165306 respectively), otherwise the average net_paid. One row, five columns bucket1..bucket5.",
 10: "Demographic profile of active customers in five counties: customers whose current address county is Rush County, Toole "
     "County, Jefferson County, Dona Ana County or La Porte County, who made a store sale in months 1-4 of 2002 and also a "
     "web sale (bill customer) or catalog sale (ship customer) in months 1-4 of 2002. Group by gender, marital status, "
     "education status, purchase estimate, credit rating, dependent count, employed-dependent count and college-dependent "
     "count; return gender, marital status, education status, count, purchase estimate, count, credit rating, count, "
     "dep count, count, dep employed count, count, dep college count, count (the same count repeated after each attribute). "
     "Order by all eight grouping columns; first 100.",
 11: "Customers whose web-sales growth from 2001 to 2002 exceeded their store-sales growth. Yearly total per customer and "
     "channel = sum of (ext_list_price - ext_discount_amt) (store via ss_customer_sk, web via ws_bill_customer_sk). Keep "
     "customers with positive 2001 totals in both channels where the web ratio (2002 total / 2001 total) is greater than "
     "the store ratio. Return c_customer_id, first name, last name and preferred-customer flag, ordered by those columns "
     "(NULLs first); first 100.",
 12: "Web sales of items in categories Sports, Books or Home sold between 1999-02-22 and 1999-03-24 inclusive: per item "
     "(id, description, category, class, current price) the item revenue (sum of ws_ext_sales_price) and its share of the "
     "class's revenue in percent (item revenue * 100 / total revenue of all returned items in the same class). Order by "
     "category, class, item id, item description, revenue ratio; first 100.",
 13: "Averages for store sales in 2001 matching one of three customer profiles and one of three address profiles. Profiles "
     "(customer_demographics joined on ss_cdemo_sk, household_demographics on ss_hdemo_sk): married ('M') with an Advanced "
     "Degree, sales price 100-150 and 3 household dependents; or single ('S') with College, sales price 50-100 and 1 "
     "dependent; or widowed ('W') with a 2 yr Degree, sales price 150-200 and 1 dependent. Address (ss_addr_sk, United "
     "States): state TX or OH with net profit 100-200; or OR, NM or KY with net profit 150-300; or VA, TX or MS with net "
     "profit 50-250. Return the average quantity, average ext_sales_price, average ext_wholesale_cost and the sum of "
     "ext_wholesale_cost (one row).",
 14: "Cross-channel best sellers in November 2001. Cross items = items whose (brand id, class id, category id) combination "
     "was sold in all three channels (store, catalog, web) during 1999-2001. Average sales = average of quantity * list "
     "price over all store, catalog and web sales in 1999-2001. For each channel ('store', 'catalog', 'web'), among sales of "
     "cross items in November 2001 grouped by brand id, class id, category id, keep groups whose sum of quantity * list "
     "price exceeds the average sales; report channel, brand id, class id, category id, the sum of that sales figure and the "
     "number of sales rows, with ROLLUP subtotals over (channel, brand id, class id, category id) and a grand total. Order "
     "by channel, brand id, class id, category id, NULLs first; first 100.",
 15: "Catalog sales in the second quarter (d_qoy = 2) of 2001 where the customer's address zip starts with one of 85669, "
     "86197, 88274, 83405, 86475, 85392, 85460, 80348, 81792, or the address state is CA, WA or GA, or the sales price "
     "exceeds 500: total cs_sales_price by customer zip (ca_zip). Order by zip, NULLs first; first 100.",
 16: "Catalog orders shipped between 2002-02-01 and 2002-04-02 to Georgia (ship address state 'GA') through call centers "
     "in Williamson County, where the order was shipped from more than one warehouse (another catalog_sales row with the "
     "same order number and a different warehouse) and has no catalog return: the count of distinct order numbers, the "
     "total ext_ship_cost and the total net profit. One row; columns \"order count\", \"total shipping cost\", "
     "\"total net profit\".",
 17: "Quantity statistics for items sold in a store in quarter '2001Q1' (d_quarter_name), returned by the same customer in "
     "2001Q1-Q3 (matched on customer, item and ticket number) and then bought again by that customer through the catalog in "
     "2001Q1-Q3 (matched on customer and item). Per item id, item description and store state: count, average, sample "
     "standard deviation and coefficient of variation (stddev / avg) of the store-sale quantity, of the return quantity and "
     "of the catalog quantity (15 columns). Order by item id, description, state (NULLs first); first 100.",
 18: "Catalog sales in 1998 billed to female customers with education status 'Unknown' (bill demographics), whose birth "
     "month is 1, 6, 8, 9, 12 or 2 and whose current address state is MS, IN, ND, OK, NM or VA: averages of quantity, list "
     "price, coupon amount, sales price, net profit, customer birth year and the bill demographics' dependent count, grouped "
     "by item id, country, state, county with ROLLUP subtotals and a grand total. Return item id, country, state, county and "
     "the seven averages, ordered by country, state, county, item id (NULLs first); first 100.",
 19: "Store sales in November 1998 of items with manager id 8, where the customer's 5-digit zip differs from the store's "
     "5-digit zip: total ext_sales_price by brand id, brand, manufacturer id and manufacturer. Order by total descending, "
     "then brand, brand id, manufacturer id, manufacturer; first 100.",
 20: "Catalog sales of items in categories Sports, Books or Home sold between 1999-02-22 and 1999-03-24 inclusive: per item "
     "(id, description, category, class, current price) the item revenue (sum of cs_ext_sales_price) and its share of the "
     "class's revenue in percent (item revenue * 100 / total revenue of returned items in the same class). Order by category, "
     "class, item id, description, revenue ratio (NULLs first); first 100.",
 21: "Inventory shift around 2000-03-11 for items priced between 0.99 and 1.49: for inventory dates 2000-02-10 to 2000-04-10, "
     "per warehouse name and item id, sum quantity on hand before 2000-03-11 (inv_before) and from 2000-03-11 on (inv_after). "
     "Keep pairs where inv_after / inv_before is between 2/3 and 3/2 (inv_before > 0). Return warehouse name, item id, "
     "inv_before, inv_after ordered by warehouse name then item id (NULLs first); first 100.",
 22: "Average quantity on hand for inventory in month sequences 1200 to 1211, grouped by product name, brand, class and "
     "category with ROLLUP subtotals and a grand total. Return product name, brand, class, category and the average, ordered "
     "by the average, then product name, brand, class, category (NULLs first); first 100.",
 23: "Big spenders buying frequent items in February 2000. Frequent items: (first 30 chars of item description, item, sold "
     "date) with more than 4 store sales during 2000-2003. Best customers: customers whose total store spend (quantity * "
     "sales price) exceeds 50 % of the maximum per-customer store spend over 2000-2003. Sum quantity * list price of catalog "
     "sales and, separately, of web sales in February 2000 by best customers for frequent items, grouped by customer last and "
     "first name (catalog rows and web rows kept as separate rows via UNION ALL). Return last name, first name, sales ordered "
     "by last name, first name, sales (NULLs first); first 100.",
 24: "Store sales that were returned (matched on ticket number and item) at stores with market id 8, where the store zip "
     "equals the customer's zip and the customer's birth country differs from the upper-cased address country: sum net "
     "paid per customer last name, first name, store name, state, item colour, price, manager, units and size. For colour "
     "'peach', total net paid by last name, first name and store name, keeping totals greater than 5 % of the average net "
     "paid across all those per-group sums. Order by last name, first name, store name.",
 25: "Items sold in a store in April 2001, returned by the same customer between April and October 2001 (customer, item, "
     "ticket), then bought again by that customer via catalog between April and October 2001 (customer, item): per item id, "
     "item description, store id and store name, the sum of store net profit, the sum of store return net loss and the sum "
     "of catalog net profit. Order by item id, description, store id, store name; first 100.",

 26: "For catalog sales in the year 2000 billed to male, single, college-educated customers (bill demographics: gender 'M', "
     "marital status 'S', education 'College') under promotions with no email channel or no event channel: per item id, the "
     "average quantity, average list price, average coupon amount and average sales price. Order by item id; first 100.",
 27: "Store sales in 2002 at Tennessee stores (s_state 'TN') to male, single, college-educated customers: average quantity, "
     "list price, coupon amount and sales price per item id and state, plus, per item id across states, the same averages with "
     "state NULL, plus one overall row with item id and state NULL. Return item id, state, a flag g_state (0 for the item-and-"
     "state rows, 1 for the rolled-up rows) and the four averages, ordered by item id then state, NULLs first; first 100.",
 28: "Six store-sales buckets by quantity: 0-5, 6-10, 11-15, 16-20, 21-25, 26-30. Within each bucket keep rows where the list "
     "price is in [B, B+10] or the coupon amount is in [C, C+1000] or the wholesale cost is in [W, W+20], with (B, C, W) = "
     "(8, 459, 57), (90, 2323, 31), (142, 12214, 79), (135, 6071, 38), (122, 836, 17), (154, 7326, 7) for buckets 1 to 6. For "
     "each bucket report the average list price, the count of list prices and the count of distinct list prices, all six "
     "buckets side by side in one row (18 columns: B1_LP, B1_CNT, B1_CNTD, ... B6_CNTD).",
 29: "Items sold in a store in September 1999, returned by the same customer between September and December 1999 (matched on "
     "customer, item, ticket number) and bought again by that customer through the catalog in 1999, 2000 or 2001 (customer, "
     "item): per item id, item description, store id and store name, the total store quantity sold, total returned quantity "
     "and total catalog quantity. Order by item id, description, store id, store name; first 100.",
 30: "Georgia customers with unusually high web returns in 2002: per returning customer and returning-address state, total "
     "web return amount for returns dated in 2002. Keep customers whose total exceeds 1.2 times the average total for that "
     "state, and whose current address state is 'GA'. Return c_customer_id, salutation, first name, last name, preferred flag, "
     "birth day, birth month, birth year, birth country, login, email address, last review date sk and the total return, "
     "ordered by all of those columns in that order, NULLs first; first 100.",
 31: "Counties where web sales grew faster than store sales in both Q1-to-Q2 and Q2-to-Q3 of 2000. Store sales per county "
     "(customer address on ss_addr_sk), quarter and year = sum of ss_ext_sales_price; web sales per county (ws_bill_addr_sk) "
     "= sum of ws_ext_sales_price. For counties with all three quarters in both channels, return county, year 2000, web "
     "Q2/Q1 ratio, store Q2/Q1 ratio, web Q3/Q2 ratio, store Q3/Q2 ratio where the web ratio exceeds the store ratio for both "
     "steps. Order by county.",
 32: "Excess discount amount: the sum of cs_ext_discount_amt for catalog sales of items from manufacturer id 977 sold between "
     "2000-01-27 and 2000-04-26 inclusive, counting only sales whose discount exceeds 1.3 times the average catalog discount "
     "for that same item over the same date range. One value, column \"excess discount amount\".",
 33: "Total sales by manufacturer for Electronics manufacturers in May 1998 to customers in GMT offset -5 addresses: for "
     "manufacturers of any Electronics item, sum ext_sales_price across store sales (address on ss_addr_sk), catalog sales "
     "(cs_bill_addr_sk) and web sales (ws_bill_addr_sk) in May 1998 where the address gmt offset is -5. Return manufacturer id "
     "and total ordered by total ascending; first 100.",
 34: "Large store tickets in Williamson County: store sales in 1999-2001 on days of month 1-3 or 25-28 by households with buy "
     "potential '>10000' or 'Unknown', at least one vehicle and dependents per vehicle above 1.2, at stores in Williamson "
     "County. Count line items per ticket number and customer; keep tickets with 15 to 20 items. Return customer last name, "
     "first name, salutation, preferred flag, ticket number and count, ordered by last name, first name, salutation, preferred "
     "flag descending, ticket number (NULLs first).",
 35: "Demographics of customers active in the first three quarters of 2002 (a store sale, and a web sale as bill customer or a "
     "catalog sale as ship customer, all with d_qoy < 4 in 2002), grouped by current address state, gender, marital status, "
     "dependent count, employed-dependent count and college-dependent count. Return state, gender, marital status, dep count, "
     "count, min/max/avg of dep count, dep employed count, count, min/max/avg of it, dep college count, count, min/max/avg of "
     "it (18 columns). Order by the six grouping columns (NULLs first); first 100.",
 36: "Gross margin hierarchy for Tennessee store sales in 2001: gross margin = sum of net profit / sum of ext sales price, by "
     "category and class, then by category alone (class NULL), then overall (both NULL), with lochierarchy 0, 1, 2 "
     "respectively. Rank each row within its parent by gross margin ascending (partition by lochierarchy and, for the "
     "category-and-class rows, by category). Return gross margin, category, class, lochierarchy and the rank, ordered by "
     "lochierarchy descending, then category for lochierarchy 0 rows, then rank (NULLs first); first 100.",
 37: "Items with current price between 68 and 98, from manufacturers 677, 940, 694 or 808, that had inventory quantity on hand "
     "between 100 and 500 on some date between 2000-02-01 and 2000-04-01 and that appear in catalog sales: distinct item id, "
     "item description and current price, ordered by item id; first 100.",
 38: "How many distinct (customer last name, first name, date) combinations appear in store sales, catalog sales (bill "
     "customer) and web sales (bill customer) alike, for month sequences 1200 to 1211? One count.",
 39: "Inventory volatility in 2001: per warehouse, item and month, the mean and sample standard deviation of quantity on hand; "
     "coefficient of variation = stdev / mean (NULL when mean is 0). Keep (warehouse, item, month) with cov > 1. Pair each "
     "January row with the February row for the same warehouse and item. Return warehouse sk, item sk, month 1, mean, cov, "
     "then the February warehouse sk, item sk, month, mean, cov, ordered by warehouse sk, item sk, month, mean, cov, February "
     "month, mean, cov (NULLs first).",
 40: "Catalog sales net of refunds around 2000-03-11 for items priced 0.99 to 1.49: for sales sold 2000-02-10 to 2000-04-10, per "
     "warehouse state and item id, sum of (sales price minus refunded cash from a matching catalog return on order number and "
     "item, 0 when none) for sold dates before 2000-03-11 (sales_before) and on or after it (sales_after). Order by state, "
     "item id; first 100.",
 42: "Store sales in November 2000 of items with manager id 1: total ext_sales_price by year, category id and category. Return "
     "d_year, category id, category, sum ordered by sum descending, then year, category id, category; first 100.",
 43: "Store sales in 2000 at stores with GMT offset -5: per store name and store id, the sum of sales price on Sundays, Mondays, "
     "Tuesdays, Wednesdays, Thursdays, Fridays and Saturdays (seven columns). Order by store name, store id, then the seven "
     "sums; first 100.",
 44: "Best and worst performing items at store 4: per item, the average net profit of store sales at store sk 4, keeping items "
     "whose average exceeds 0.9 times the average net profit of store-4 sales with a NULL address sk. Rank items ascending "
     "and descending by that average; for ranks 1 to 10 return the rank, the product name of the item at that rank in the "
     "ascending order (best_performing) and in the descending order (worst_performing). Order by rank; first 100.",
 45: "Web sales in Q2 2001 where the customer's zip starts with one of 85669, 86197, 88274, 83405, 86475, 85392, 85460, 80348, "
     "81792, or the item id is one of the item ids of item sks 2, 3, 5, 7, 11, 13, 17, 19, 23, 29: sum of ws_sales_price by "
     "customer zip and city. Order by zip, city; first 100.",
 46: "Weekend store tickets in Fairview or Midway stores during 1999-2001 by households with 4 dependents or 3 vehicles (d_dow "
     "6 or 0): per ticket number, customer, address and the address city (bought_city), the sum of coupon amount and of net "
     "profit. Keep tickets where the customer's current address city differs from bought_city. Return last name, first name, "
     "current city, bought city, ticket number, coupon total, profit total, ordered by last name, first name, current city, "
     "bought city, ticket number (NULLs first); first 100.",
 47: "Monthly store sales by category, brand, store name and company name for Dec 1998 through Jan 2000: per month, the sum of "
     "sales price, the average monthly sum within the same year (window over the group and year), and the month's rank in "
     "the group's chronological order. For 1999 months whose sum deviates from the year's average by more than 10 % (average "
     "> 0), return category, brand, store name, company name, year, month, average monthly sales, the month's sum, the "
     "previous month's sum and the next month's sum (adjacent ranks in the same group). Order by (sum minus average), then "
     "the ten output columns in order; first 100.",
 48: "Total store-sales quantity in 2000 for sales matching one of three customer profiles (married 'M' with a 4 yr Degree "
     "and sales price 100-150; divorced 'D' with a 2 yr Degree and 50-100; single 'S' with College and 150-200) and one of "
     "three United States address profiles on ss_addr_sk (state CO, OH or TX with net profit 0-2000; OR, MN or KY with "
     "150-3000; VA, CA or MS with 50-25000). One value.",
 49: "Worst return ratios by channel in December 2001. For each channel (web, catalog, store), per item: return ratio = sum of "
     "return quantity / sum of sold quantity and currency ratio = sum of return amount / sum of net paid, over sales in "
     "December 2001 left-joined to their returns (web: order number and item; catalog: order number and item; store: ticket "
     "number and item), keeping only sales with a matching return amount above 10000, net profit > 1, net paid > 0 and "
     "quantity > 0. Rank items per channel by return ratio and by currency ratio ascending; keep items in the top 10 of "
     "either. Return channel, item sk, return ratio, return rank, currency rank (distinct rows), ordered by channel, return "
     "rank, currency rank, item (NULLs first); first 100.",
 50: "Return latency by store for store returns in August 2001 (return date), matched to their sale on ticket number, item "
     "and customer: per store (name, company id, street number, street name, street type, suite number, city, county, "
     "state, zip) count returns made within 30 days of the sale date (difference of date surrogate keys), 31-60 days, 61-90 "
     "days, 91-120 days and over 120 days (five columns named \"30 days\", \"31-60 days\", \"61-90 days\", \"91-120 days\", "
     "\">120 days\"). Order by the ten store columns; first 100.",

 51: "Days when an item's cumulative web sales overtook its cumulative store sales, for month sequences 1200-1211: per item "
     "and date, the running total of ws_sales_price (web) and of ss_sales_price (store) ordered by date within the item; full-"
     "outer-join web and store by item and date, and carry each side's running maximum forward over dates (max over rows up "
     "to the current one). Return item sk, date, web running total, store running total, web cumulative max, store cumulative "
     "max where the web cumulative exceeds the store cumulative. Order by item sk, date (NULLs first); first 100.",
 52: "Store sales in November 2000 of items with manager id 1: total ext_sales_price by year, brand id and brand. Return "
     "d_year, brand id, brand, total ordered by year, total descending, brand id; first 100.",
 53: "Quarterly store sales by manufacturer that deviate from the manufacturer's quarterly average by more than 10 %, for "
     "month sequences 1200-1211 and items in either profile: category Books, Children or Electronics with class personal, "
     "portable, reference or self-help and brand scholaramalgamalg #14, scholaramalgamalg #7, exportiunivamalg #9 or "
     "scholaramalgamalg #9; or category Women, Music or Men with class accessories, classical, fragrances or pants and brand "
     "amalgimporto #1, edu packscholar #1, exportiimporto #1 or importoamalg #1. Per manufacturer id and quarter (d_qoy): the "
     "sum of sales price and the average of those quarterly sums for the manufacturer. Return manufacturer id, quarterly sum, "
     "average where average > 0 and |sum - average| / average > 0.1, ordered by average, sum, manufacturer id; first 100.",
 54: "Revenue segments of December 1998 maternity buyers: customers who bought a Women / maternity item via catalog or web "
     "(bill customer) in December 1998. For those customers, sum ss_ext_sales_price of store sales in the three month "
     "sequences after December 1998 (month_seq+1 to month_seq+3) at stores in the same county and state as the customer's "
     "current address. Segment = round(revenue / 50) as an integer. Return segment, number of customers, segment * 50, "
     "ordered by segment, count, segment base (NULLs first); first 100.",
 55: "Store sales in November 1999 of items with manager id 28: total ext_sales_price by brand id and brand, ordered by "
     "total descending then brand id; first 100.",
 56: "Total sales in February 2001 of items whose colour is slate, blanched or burnished (by item id, i.e. all item sks "
     "sharing the id), to addresses with GMT offset -5 (store: ss_addr_sk; catalog: cs_bill_addr_sk; web: ws_bill_addr_sk): "
     "sum ext_sales_price across the three channels per item id. Order by total then item id (NULLs first); first 100.",
 57: "Monthly catalog sales by category, brand and call center name for Dec 1998 through Jan 2000: per month, the sum of "
     "sales price, the average monthly sum within the same year (window over the group and year), and the month's rank in "
     "the group's chronological order. For 1999 months whose sum deviates from the year's average by more than 10 % (average "
     "> 0), return category, brand, call center name, year, month, average monthly sales, the month's sum, the previous "
     "month's sum and the next month's sum (adjacent ranks in the same group). Order by (sum minus average) NULLs first, then "
     "the nine output columns in order; first 100.",
 58: "Items with balanced channel revenue in the week of 2000-01-03: for the dates of that week (same d_week_seq), per item "
     "id, the store revenue (sum ss_ext_sales_price), catalog revenue (cs_ext_sales_price) and web revenue "
     "(ws_ext_sales_price). Keep items where each channel's revenue is within 90 %-110 % of each of the other two. Return "
     "item id, store revenue, store revenue as a percent of the three-channel average (rev / avg * 100), catalog revenue and "
     "its percent, web revenue and its percent, and the average. Order by item id, store revenue (NULLs first); first 100.",
 59: "Year-over-year weekday sales ratios per store: per week (d_week_seq) and store, the sum of ss_sales_price on Sundays "
     "through Saturdays. Pair weeks in month sequences 1212-1223 (year 1) with the week 52 later in month sequences 1224-1235 "
     "(year 2) for the same store id. Return store name, store id, the year-1 week sequence and the seven year-1 / year-2 "
     "ratios (Sunday through Saturday). Order by store name, store id, week (NULLs first); first 100.",
 60: "Total sales in September 1998 of Music items (by item id) to addresses with GMT offset -5 (store ss_addr_sk, catalog "
     "cs_bill_addr_sk, web ws_bill_addr_sk): sum ext_sales_price across the three channels per item id. Order by item id then "
     "total; first 100.",
 61: "Promotion share for Jewelry in November 1998, GMT offset -5 stores and customers: promotional sales = total "
     "ss_ext_sales_price of store sales of Jewelry items in November 1998 at stores with GMT offset -5 to customers whose "
     "address GMT offset is -5, under promotions with direct mail, email or TV channel = 'Y'; total = the same without the "
     "promotion condition. Return promotions, total and promotions / total * 100 (one row).",
 62: "Web shipping latency by warehouse, ship mode and site for ship dates in month sequences 1200-1211: per first 20 "
     "characters of the warehouse name, ship mode type and web site name, count sales shipped within 30 days of the sold "
     "date (difference of date surrogate keys), 31-60, 61-90, 91-120 and over 120 days (columns \"30 days\", \"31-60 days\", "
     "\"61-90 days\", \"91-120 days\", \">120 days\"). Order by the three grouping columns (NULLs first); first 100.",
 63: "Monthly store sales by manager that deviate from the manager's monthly average by more than 10 %, for month sequences "
     "1200-1211 and items in either profile: category Books, Children or Electronics with class personal, portable, "
     "reference or self-help and brand scholaramalgamalg #14, scholaramalgamalg #7, exportiunivamalg #9 or "
     "scholaramalgamalg #9; or category Women, Music or Men with class accessories, classical, fragrances or pants and brand "
     "amalgimporto #1, edu packscholar #1, exportiimporto #1 or importoamalg #1. Per manager id and month (d_moy): the sum "
     "of sales price and the average of those monthly sums for the manager. Return manager id, monthly sum, average where "
     "average > 0 and |sum - average| / average > 0.1, ordered by manager id, average, sum; first 100.",
 65: "Slow-selling items per store for month sequences 1176-1187: revenue = sum of ss_sales_price per store and item; store "
     "average = average of those item revenues within the store. For items whose revenue is at most 10 % of their store's "
     "average, return store name, item description, the item's revenue, current price, wholesale cost and brand. Order by "
     "store name, item description (NULLs first); first 100.",
 67: "Top-100 sales groups per category with ROLLUP for month sequences 1200-1211: sum of ss_sales_price * ss_quantity (0 when "
     "NULL) grouped by ROLLUP over (category, class, brand, product name, year, quarter, month, store id); rank rows within "
     "each category by that sum descending and keep rank <= 100. Return the eight grouping columns, the sum and the rank, "
     "ordered by all ten columns (NULLs first); first 100.",
 68: "Store tickets on the 1st or 2nd of the month in 1999-2001 at Fairview or Midway stores by households with 4 dependents "
     "or 3 vehicles: per ticket, customer, address and address city (bought_city), the sums of ext_sales_price, "
     "ext_list_price and ext_tax. Keep tickets where the customer's current city differs from bought_city. Return last name, "
     "first name, current city, bought city, ticket number, extended price sum, extended tax sum, list price sum, ordered by "
     "last name then ticket number (NULLs first); first 100.",
 69: "Demographics of store-only shoppers in Kentucky, Georgia and New Mexico: customers with current address state KY, GA "
     "or NM who made a store sale in April-June 2001 and made no web sale (bill customer) and no catalog sale (ship customer) "
     "in April-June 2001. Group by gender, marital status, education status, purchase estimate, credit rating; return gender, "
     "marital status, education status, count, purchase estimate, count, credit rating, count (the same count three times). "
     "Order by the five grouping columns; first 100.",
 70: "Net profit hierarchy for the top states, month sequences 1200-1211: candidate states are those whose store net profit "
     "sum ranks in the top 5 within the state (i.e. every state with sales, as ranked per state). Sum ss_net_profit by state "
     "and county with ROLLUP subtotals per state and a grand total; lochierarchy = number of rolled-up columns (0, 1, 2); rank "
     "rows within their parent by the sum descending (partition by lochierarchy and, for county rows, the state). Return the "
     "sum, state, county, lochierarchy and the rank, ordered by lochierarchy descending, then state for county-level rows, "
     "then rank; first 100.",
 71: "Breakfast and dinner sales of manager-1 items in November 1999 across all three channels (web, catalog, store): sum of "
     "ext_sales_price by brand id, brand, sale hour and minute (time_dim), for meal times 'breakfast' or 'dinner'. Return brand "
     "id, brand, hour, minute, total ordered by total descending, brand id, hour (NULLs first).",
 72: "Catalog orders at risk of stock-out in 1999: catalog sales (sold in 1999) to households with buy potential '>10000' by "
     "divorced customers (cd_marital_status 'D'), joined to inventory of the same item in the same week (d_week_seq of the "
     "inventory date equals that of the sold date) where quantity on hand is below the ordered quantity, and shipped more "
     "than 5 days after the sold date; left-join promotion and catalog returns. Per item description, warehouse name and "
     "sold-week sequence: count of rows without a promotion, with a promotion, and total. Order by total descending, item "
     "description, warehouse name, week (NULLs first); first 100.",
 73: "Small store tickets (1 to 5 line items) on the 1st or 2nd of the month in 1999-2001 at stores in Orange County, Bronx "
     "County, Franklin Parish or Williamson County, by households with buy potential 'Unknown' or '>10000', at least one "
     "vehicle and dependents per vehicle above 1: per ticket and customer the line count. Return last name, first name, "
     "salutation, preferred flag, ticket number, count ordered by count descending then last name.",
 74: "Customers whose web net paid grew faster than their store net paid from 2001 to 2002: yearly total per customer and "
     "channel = sum of net paid (store ss_net_paid via ss_customer_sk, web ws_net_paid via ws_bill_customer_sk) for 2001 and "
     "2002. Keep customers with positive 2001 totals in both channels where web 2002/2001 exceeds store 2002/2001. Return "
     "c_customer_id, first name, last name ordered by customer id (NULLs first); first 100.",
 75: "Books groups whose net unit sales fell more than 10 % from 2001 to 2002: net sales per row = quantity minus returned "
     "quantity and ext_sales_price minus return amount (returns matched on order number and item for catalog and web, ticket "
     "and item for store; 0 when none), over Books items in all three channels combined with UNION (distinct rows), summed by "
     "year, brand id, class id, category id, manufacturer id. For groups where 2002 count / 2001 count < 0.9, return previous "
     "year, year, brand id, class id, category id, manufacturer id, previous count, current count, count difference and "
     "amount difference. Order by count difference then amount difference; first 100.",

 76: "Sales with missing keys by channel: store sales with a NULL store sk, web sales with a NULL ship customer sk and "
     "catalog sales with a NULL ship address sk. Per channel ('store', 'web', 'catalog'), the name of the null column "
     "('ss_store_sk', 'ws_ship_customer_sk', 'cs_ship_addr_sk'), year, quarter and item category: the row count and the sum "
     "of ext_sales_price. Order by channel, column name, year, quarter, category (NULLs first); first 100.",
 77: "Sales, returns and profit by channel for 2000-08-23 to 2000-09-22 inclusive: store channel per store sk (sales = sum "
     "ss_ext_sales_price, returns = sum sr_return_amt by return date, 0 when none, profit = sum ss_net_profit minus sum "
     "sr_net_loss); catalog channel per call center sk with cs_ext_sales_price / cr_return_amount / cs_net_profit minus "
     "cr_net_loss, where every catalog sales group is paired with every catalog returns group (cross join); web channel per "
     "web page sk with ws_ext_sales_price / wr_return_amt / ws_net_profit minus wr_net_loss (left join). Return channel, id, "
     "sales, returns, profit with ROLLUP subtotals per channel and a grand total, ordered by channel, id (NULLs first), "
     "returns descending; first 100.",
 78: "Store-loyal item purchases in 2000: for sales with no return (store: no matching store return on ticket and item; web "
     "and catalog: no matching return on order number and item), sum quantity, wholesale cost and sales price per year, item "
     "and customer in each channel (web and catalog by bill customer). For 2000 store groups that also bought the item via "
     "web or catalog (other-channel quantity > 0), return year, item sk, customer sk, store quantity / other-channel "
     "quantity rounded to 2 decimals, store quantity, store wholesale cost, store sales price, other-channel quantity, "
     "wholesale cost and sales price (web + catalog, 0 when absent). Order by year, item, customer, store qty desc, store "
     "wholesale cost desc, store sales price desc, other qty, other wholesale cost, other sales price, ratio; first 100.",
 79: "Monday store tickets in 1999-2001 (d_dow = 1) at stores with 200 to 295 employees, by households with 6 dependents or "
     "more than 2 vehicles: per ticket, customer, address and store city, the sums of coupon amount and net profit. Return "
     "customer last name, first name, the first 30 characters of the store city, ticket number, coupon total, profit total, "
     "ordered by last name, first name, city, profit (NULLs first), ticket number; first 100.",
 80: "Sales, returns and profit by channel for 2000-08-23 to 2000-09-22, items priced above 50, promotions without a TV "
     "channel (p_channel_tv = 'N'): store channel per store id (id = 'store' || s_store_id): sales = sum ss_ext_sales_price, "
     "returns = sum of matched sr_return_amt (ticket and item, 0 when none), profit = sum of ss_net_profit minus matched "
     "sr_net_loss; catalog channel per catalog page ('catalog_page' || id) with cs_* and cr_* matched on order and item; web "
     "channel per web site ('web_site' || id) with ws_* and wr_* matched on order and item. Return channel, id, sales, "
     "returns, profit with ROLLUP subtotals per channel and a grand total, ordered by channel, id (NULLs first); first 100.",
 81: "Georgia customers with unusually high catalog returns in 2000: per returning customer and returning-address state, "
     "the total cr_return_amt_inc_tax for returns dated in 2000. Keep customers whose total exceeds 1.2 times the average "
     "total for that state and whose current address state is 'GA'. Return c_customer_id, salutation, first name, last name, "
     "street number, street name, street type, suite number, city, county, state, zip, country, gmt offset, location type "
     "and the total, ordered by all those columns in order; first 100.",
 82: "Items with current price between 62 and 92, from manufacturers 129, 270, 821 or 423, with inventory quantity on hand "
     "between 100 and 500 on some date between 2000-05-25 and 2000-07-24, that appear in store sales: distinct item id, "
     "description and current price ordered by item id; first 100.",
 83: "Return quantities by channel for the weeks containing 2000-06-30, 2000-09-27 and 2000-11-17: per item id, the total "
     "returned quantity in store returns, catalog returns and web returns (return dates in those weeks). For items present in "
     "all three, return item id, store qty, store qty / (sum of the three) / 3 * 100, catalog qty, its same ratio, web qty, "
     "its ratio, and the three-channel average (sum / 3). Order by item id, store qty (NULLs first); first 100.",
 84: "Customers in Edgewood whose household income band lies between 38128 and 88128 (lower bound >= 38128, upper bound <= "
     "88128) and whose current demographics appear as the returning demographics of a store return (sr_cdemo_sk): return "
     "c_customer_id and 'last name, first name' (NULL names as empty), one row per matching store return, ordered by customer "
     "id (NULLs first); first 100.",
 85: "Web returns in 2000 (by sold date) matched to their sale on item and order number, joined to web page, the refunded and "
     "returning customer demographics, the refunded address and the reason: keep rows where the refunded and returning "
     "demographics share marital status and education and match one profile (married 'M' with an Advanced Degree and sales "
     "price 100-150; single 'S' with College and 50-100; widowed 'W' with a 2 yr Degree and 150-200), and the refunded address "
     "is in the United States in IN, OH or NJ with net profit 100-200, or WI, CT or KY with 150-300, or LA, IA or AR with "
     "50-250. Per reason description: its first 20 characters, average quantity, average refunded cash and average fee, "
     "ordered by those four columns; first 100.",
 86: "Web net paid hierarchy for month sequences 1200-1211: sum of ws_net_paid by item category and class with ROLLUP "
     "subtotals per category and a grand total; lochierarchy = number of rolled-up columns (0, 1, 2); rank rows within their "
     "parent by the sum descending (partition by lochierarchy and, for class rows, the category). Return sum, category, class, "
     "lochierarchy, rank ordered by lochierarchy descending, then category for class-level rows, then rank (NULLs first); "
     "first 100.",
 87: "How many distinct (customer last name, first name, date) combinations appear in store sales during month sequences "
     "1200-1211 but in neither catalog sales (bill customer) nor web sales (bill customer) over the same months? One count.",
 88: "Store 'ese' morning traffic by half hour for households with (4 dependents and at most 6 vehicles) or (2 dependents and "
     "at most 4 vehicles) or (0 dependents and at most 2 vehicles): count store sales at stores named 'ese' in each of the "
     "half-hour slots 8:30-9:00, 9:00-9:30, 9:30-10:00, 10:00-10:30, 10:30-11:00, 11:00-11:30, 11:30-12:00, 12:00-12:30 (by "
     "t_hour and t_minute). One row with eight columns h8_30_to_9, h9_to_9_30, h9_30_to_10, h10_to_10_30, h10_30_to_11, "
     "h11_to_11_30, h11_30_to_12, h12_to_12_30.",
 89: "Monthly store sales in 1999 that deviate from the group's monthly average by more than 10 %, for items in categories "
     "Books, Electronics or Sports with class computers, stereo or football, or categories Men, Jewelry or Women with class "
     "shirts, birdal or dresses. Group by category, class, brand, store name, company name and month: the sum of sales price "
     "and the average of those monthly sums over the (category, brand, store name, company name) group. Return category, "
     "class, brand, store name, company name, month, sum, average where average <> 0 and |sum - average| / average > 0.1, "
     "ordered by (sum minus average), store name, category, class, brand, company name, month, sum, average; first 100.",
 90: "Morning-to-evening ratio of web sales: count web sales sold between 8:00 and 9:59 (t_hour 8 or 9) and between 19:00 and "
     "20:59 (t_hour 19 or 20), for ship households with 6 dependents on web pages with 5000 to 5200 characters "
     "(wp_char_count). Return the morning count divided by the evening count (NULL if the evening count is 0), one value.",
 91: "Call-center return losses in November 1998 from customers with GMT offset -7 addresses, household buy potential "
     "starting with 'Unknown', and demographics married 'M' with education 'Unknown' or widowed 'W' with an Advanced Degree: "
     "per call center id, name and manager (and demographic combination), the sum of cr_net_loss. Return call center id, "
     "name, manager, loss ordered by loss descending.",
 92: "Excess web discount: the sum of ws_ext_discount_amt for web sales of items from manufacturer id 350 sold between "
     "2000-01-27 and 2000-04-26 inclusive, counting only sales whose discount exceeds 1.3 times the average web discount for "
     "that same item over the same date range. One value, column \"Excess Discount Amount\".",
 93: "Actual sales per customer after returns for reason 'reason 28': for store sales left-joined to store returns (item and "
     "ticket) where the return reason is 'reason 28', actual sales per line = (quantity minus returned quantity) * sales "
     "price when returned, else quantity * sales price. Sum per customer sk; return customer sk and the sum ordered by sum "
     "then customer sk (NULLs first); first 100.",
 94: "Web orders shipped between 1999-02-01 and 1999-04-02 to Illinois (ship address state 'IL') through web sites of company "
     "'pri', shipped from more than one warehouse (another web_sales row with the same order number and a different "
     "warehouse) and with no web return: the count of distinct order numbers, total ext_ship_cost and total net profit. One "
     "row; columns \"order count\", \"total shipping cost\", \"total net profit\".",
 95: "Web orders shipped between 1999-02-01 and 1999-04-02 to Illinois through web sites of company 'pri', shipped from more "
     "than one warehouse and that do have a web return: the count of distinct order numbers, total ext_ship_cost and total "
     "net profit. One row; columns \"order count\", \"total shipping cost\", \"total net profit\".",
 96: "How many store sales happened at stores named 'ese' between 20:30 and 20:59 (t_hour 20, t_minute >= 30) to households "
     "with 7 dependents? One count.",
 97: "Customer-item pairs by channel for month sequences 1200-1211: distinct (customer, item) pairs from store sales and, "
     "separately, from catalog sales (bill customer). Full outer join them on customer and item and count pairs that are "
     "store-only, catalog-only and in both. One row, three columns store_only, catalog_only, store_and_catalog.",
 98: "Store sales of items in categories Sports, Books or Home sold between 1999-02-22 and 1999-03-24 inclusive: per item "
     "(id, description, category, class, current price) the item revenue (sum of ss_ext_sales_price) and its share of the "
     "class's revenue in percent (item revenue * 100 / total revenue of returned items in the same class). Order by category, "
     "class, item id, description, revenue ratio (NULLs first). No row limit.",
 99: "Catalog shipping latency by warehouse, ship mode and call center for ship dates in month sequences 1200-1211: per first "
     "20 characters of the warehouse name, ship mode type and lower-cased call center name, count sales shipped within 30 "
     "days of the sold date (difference of date surrogate keys), 31-60, 61-90, 91-120 and over 120 days (columns \"30 days\", "
     "\"31-60 days\", \"61-90 days\", \"91-120 days\", \">120 days\"). Order by the three grouping columns (NULLs first); "
     "first 100.",
}
