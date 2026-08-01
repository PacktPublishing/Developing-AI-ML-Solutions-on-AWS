"""Pipeline step 4 (Processing): apply the limit decisions to the warehouse."""

import csv
import json
import os

import psycopg2

IN = "/opt/ml/processing/input"
OUT = "/opt/ml/processing/output"


def main() -> None:
    """Update the limits that change; write a run summary."""
    with open(f"{IN}/decisions.csv") as f:
        changes = [
            (row["new_limit"], row["customer_id"])
            for row in csv.DictReader(f)
            if row["decision"] != "KEEP"
        ]

    with psycopg2.connect(os.environ["WAREHOUSE_DSN"]) as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE customers SET current_limit = %s WHERE customer_id = %s", changes
        )

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"applied": len(changes)}, f)
    print(f"applied {len(changes)} limit changes")


if __name__ == "__main__":
    main()
