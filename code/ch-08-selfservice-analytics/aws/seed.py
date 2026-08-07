# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Load the credit mart into the Redshift Serverless workgroup over the Data API.

Runs local/seed/analytics.sql (the same seed the local stack mounts) as batched
statements against the deployed workgroup. Env: REDSHIFT_WORKGROUP, REDSHIFT_DB_NAME.
"""

import os
import pathlib
import time

import boto3

REGION = os.environ.get("REDSHIFT_REGION", "us-east-1")
WORKGROUP = os.environ["REDSHIFT_WORKGROUP"]
DB = os.environ["REDSHIFT_DB_NAME"]

sql_file = (
    pathlib.Path(__file__).resolve().parents[1] / "local" / "seed" / "analytics.sql"
)
# strip full-line comments, then split into statements the batch API runs in order
body = "\n".join(
    line
    for line in sql_file.read_text().splitlines()
    if not line.strip().startswith("--")
)
statements = [s.strip() for s in body.split(";") if s.strip()]

client = boto3.client("redshift-data", region_name=REGION)
started = client.batch_execute_statement(
    WorkgroupName=WORKGROUP, Database=DB, Sqls=statements
)
while True:
    desc = client.describe_statement(Id=started["Id"])
    if desc["Status"] in ("FINISHED", "FAILED", "ABORTED"):
        break
    time.sleep(1)

print(f"seed: {desc['Status']} ({len(statements)} statements)")
if desc["Status"] != "FINISHED":
    raise SystemExit(desc.get("Error", "seed failed"))
