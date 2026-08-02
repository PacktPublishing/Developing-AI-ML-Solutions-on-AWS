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


def main() -> None:
    """Read the staged decisions and update the limits that change."""
    s3 = s3_client()
    staged = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/decisions.csv")["Body"].read()
    changes = [
        (row["new_limit"], row["customer_id"])
        for row in csv.DictReader(io.StringIO(staged.decode()))
        if row["decision"] != "KEEP"
    ]

    with psycopg2.connect(WAREHOUSE_DSN) as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE customers SET current_limit = %s WHERE customer_id = %s", changes
        )
    print(f"applied {len(changes)} limit changes")


if __name__ == "__main__":
    main()
