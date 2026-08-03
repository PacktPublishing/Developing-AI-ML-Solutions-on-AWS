"""Pipeline step 4 (Processing): apply the limit decisions to the warehouse."""

import csv
import json
import os

import psycopg2
from psycopg2.extras import execute_values

IN = "/opt/ml/processing/input"
OUT = "/opt/ml/processing/output"


def main() -> None:
    """Update the limits that change; write a run summary."""
    with open(f"{IN}/decisions.csv") as f:
        changes = [
            (row["customer_id"], row["new_limit"])
            for row in csv.DictReader(f)
            if row["decision"] != "KEEP"
        ]

    # Stage the changes and apply them in one set-based UPDATE ... FROM. Redshift
    # rewrites blocks on every single-row UPDATE, so 3,800 of them crawl; one
    # UPDATE joined to a staged table is a single pass. Runs the same on Postgres.
    with psycopg2.connect(os.environ["WAREHOUSE_DSN"]) as conn, conn.cursor() as cur:
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

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"applied": len(changes)}, f)
    print(f"applied {len(changes)} limit changes")


if __name__ == "__main__":
    main()
