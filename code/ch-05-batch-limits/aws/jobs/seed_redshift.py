# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Seed the Redshift Serverless warehouse through the Data API.

Env: WORKGROUP, BUCKET, COPY_ROLE_ARN (required), DB_NAME (default portfolio).
"""

import os
import time

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
WORKGROUP = os.environ["WORKGROUP"]
DB = os.environ.get("DB_NAME", "portfolio")
BUCKET = os.environ["BUCKET"]
COPY_ROLE = os.environ["COPY_ROLE_ARN"]
KEY = "ch05/seed/portfolio.csv"

CREATE = """
CREATE TABLE customers (
    customer_id        VARCHAR(16),
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


def run(client, sql: str) -> None:
    """Run one statement through the Data API and wait for it."""
    sid = client.execute_statement(WorkgroupName=WORKGROUP, Database=DB, Sql=sql)["Id"]
    while True:
        desc = client.describe_statement(Id=sid)
        if desc["Status"] == "FINISHED":
            return
        if desc["Status"] in ("FAILED", "ABORTED"):
            raise RuntimeError(desc.get("Error", desc["Status"]))
        time.sleep(1)


def main() -> None:
    """Stage the CSV and rebuild + COPY the customers table."""
    boto3.client("s3", region_name=REGION).upload_file(
        "data/portfolio.csv", BUCKET, KEY
    )
    print(f"uploaded s3://{BUCKET}/{KEY}")

    rc = boto3.client("redshift-data", region_name=REGION)
    run(rc, "DROP TABLE IF EXISTS customers")
    run(rc, CREATE)
    run(
        rc,
        f"COPY customers FROM 's3://{BUCKET}/{KEY}' "
        f"IAM_ROLE '{COPY_ROLE}' CSV IGNOREHEADER 1 REGION '{REGION}'",
    )
    print("customers loaded")


if __name__ == "__main__":
    main()
