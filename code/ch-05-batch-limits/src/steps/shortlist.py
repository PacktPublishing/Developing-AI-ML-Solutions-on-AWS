# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3", "psycopg2-binary"]
# ///
"""Shortlist the customers eligible for a limit review.

Usage:
  uv run steps/shortlist.py
"""

import csv
import io

import psycopg2
from common import BUCKET, PREFIX, WAREHOUSE_DSN, s3_client

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
    """Query the eligible book and stage it to S3."""
    with psycopg2.connect(WAREHOUSE_DSN) as conn, conn.cursor() as cur:
        cur.execute(ELIGIBLE_SQL)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()

    body = io.StringIO()
    writer = csv.writer(body)
    writer.writerow(columns)
    writer.writerows(rows)
    key = f"{PREFIX}/eligible.csv"
    s3_client().put_object(Bucket=BUCKET, Key=key, Body=body.getvalue().encode())
    print(f"{len(rows)} eligible customers -> s3://{BUCKET}/{key}")


if __name__ == "__main__":
    main()
