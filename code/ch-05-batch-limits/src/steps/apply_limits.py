# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3", "psycopg2-binary"]
# ///
"""Apply the staged limit decisions to the warehouse.

Usage:
  uv run steps/apply_limits.py
"""

import csv
import io

import psycopg2
from common import BUCKET, PREFIX, WAREHOUSE_DSN, s3_client
from psycopg2.extras import execute_values


def main() -> None:
    """Read the staged decisions and update the limits that change."""
    s3 = s3_client()
    staged = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/decisions.csv")["Body"].read()
    changes = [
        (row["customer_id"], row["new_limit"])
        for row in csv.DictReader(io.StringIO(staged.decode()))
        if row["decision"] != "KEEP"
    ]

    # one set-based UPDATE ... FROM a staged table, not a row-by-row UPDATE:
    # Redshift crawls on single-row UPDATEs; this stays fast there and on Postgres.
    with psycopg2.connect(WAREHOUSE_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE changes (customer_id VARCHAR(16), new_limit NUMERIC(12, 2))"
        )
        execute_values(
            cur, "INSERT INTO changes (customer_id, new_limit) VALUES %s", changes
        )
        cur.execute(
            "UPDATE customers SET current_limit = c.new_limit "
            "FROM changes c WHERE customers.customer_id = c.customer_id"
        )
    print(f"applied {len(changes)} limit changes")


if __name__ == "__main__":
    main()
