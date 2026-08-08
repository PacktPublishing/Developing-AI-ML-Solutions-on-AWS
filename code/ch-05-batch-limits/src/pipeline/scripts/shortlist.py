"""Pipeline step 1 (Processing): shortlist and history from the warehouse.

Writes eligible.csv (the book to score) and history.csv (the labeled past
the training step fits on) to two processing outputs.
"""

import csv
import os

import psycopg2

ELIGIBLE_OUT = "/opt/ml/processing/eligible"
HISTORY_OUT = "/opt/ml/processing/history"

HISTORY_SQL = """
SELECT current_limit, utilization, months_on_book, dpd_30_count_12m,
       monthly_income, payment_ratio, cash_advance_share, default_12m
FROM customers
WHERE default_12m IS NOT NULL
"""

ELIGIBLE_SQL = """
SELECT customer_id, current_limit, utilization, months_on_book,
       dpd_30_count_12m, monthly_income, payment_ratio, cash_advance_share
FROM customers
WHERE months_on_book >= 6
  AND dpd_30_count_12m <= 2
  AND utilization > 0.05
ORDER BY customer_id
"""


def dump(cur, sql: str, path: str) -> int:
    """Run one query and write the result as CSV."""
    cur.execute(sql)
    columns = [d[0] for d in cur.description]
    rows = cur.fetchall()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    """Query the eligible book and the labeled history into two channels."""
    with psycopg2.connect(os.environ["WAREHOUSE_DSN"]) as conn, conn.cursor() as cur:
        n_eligible = dump(cur, ELIGIBLE_SQL, f"{ELIGIBLE_OUT}/eligible.csv")
        n_history = dump(cur, HISTORY_SQL, f"{HISTORY_OUT}/history.csv")
    print(f"{n_eligible} eligible, {n_history} labeled")


if __name__ == "__main__":
    main()
