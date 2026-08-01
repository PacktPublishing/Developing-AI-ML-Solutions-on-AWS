# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "psycopg2-binary"]
# ///
"""Load the portfolio into the warehouse (rebuilds the customers table).

Usage:
  uv run steps/seed_warehouse.py
"""

import psycopg2
from common import WAREHOUSE_DSN

DDL = """
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
    customer_id        VARCHAR(16) PRIMARY KEY,
    current_limit      NUMERIC(12, 2),
    utilization        DOUBLE PRECISION,
    months_on_book     INTEGER,
    dpd_30_count_12m   INTEGER,
    monthly_income     NUMERIC(12, 2),
    payment_ratio      DOUBLE PRECISION,
    cash_advance_share DOUBLE PRECISION,
    default_12m        DOUBLE PRECISION
)
"""


def main() -> None:
    """Rebuild and load the customers table."""
    with psycopg2.connect(WAREHOUSE_DSN) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        with open("data/portfolio.csv") as f:
            cur.copy_expert(
                "COPY customers FROM STDIN WITH (FORMAT csv, HEADER true)", f
            )
        cur.execute("SELECT COUNT(*) FROM customers")
        print(f"customers loaded: {cur.fetchone()[0]}")


if __name__ == "__main__":
    main()
