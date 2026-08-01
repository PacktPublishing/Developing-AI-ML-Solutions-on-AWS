"""Pipeline step 1 (Processing): shortlist eligible customers from the warehouse.

Writes eligible.csv to the processing output; downstream steps read it by
S3 property reference.
"""

import csv
import os

import psycopg2

OUT = "/opt/ml/processing/output"

ELIGIBLE_SQL = """
SELECT customer_id, current_limit, utilization, months_on_book,
       dpd_30_count_12m, monthly_income, payment_ratio, cash_advance_share
FROM customers
WHERE months_on_book >= 6
  AND dpd_30_count_12m <= 2
  AND utilization > 0.05
ORDER BY customer_id
"""


def main() -> None:
    """Query the eligible book into the step's output channel."""
    with psycopg2.connect(os.environ["WAREHOUSE_DSN"]) as conn, conn.cursor() as cur:
        cur.execute(ELIGIBLE_SQL)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/eligible.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"{len(rows)} eligible customers")


if __name__ == "__main__":
    main()
