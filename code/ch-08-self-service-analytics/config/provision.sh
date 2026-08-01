#!/usr/bin/env bash
# -------------------------------------------------------------------------------
# Provision the warehouse side of the security model. Idempotent.
#
#   1. A read-only service account: bi_analyst, PASSWORD DISABLE, so the only
#      way in is temporary IAM credentials (redshift-serverless:GetCredentials,
#      the task role's path). GRANT USAGE + SELECT on analytics, nothing else:
#      read-only by construction, not by prompt.
#   2. Dynamic data masking policies generated from config/pii_columns.yaml,
#      the SAME file the helper's result masking reads, so the two enforcement
#      layers cannot drift. The policies attach to bi_analyst: even hand-written
#      SQL through the Data API comes back masked at the engine.
#
# Redshift-native DDM does not exist in redshift-local; locally the helper
# layer carries the masking, and this script belongs to the cloud round.
#
# Requires: REDSHIFT_WORKGROUP, REDSHIFT_DB_NAME, AWS credentials.
# Untested scaffold: run against the deployed workgroup before the chapter ships.
# -------------------------------------------------------------------------------

set -euo pipefail

uv run --with boto3 python3 -u - <<'PYEOF'
import os
import re
import time

import boto3

REGION = os.environ.get("REDSHIFT_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
WORKGROUP = os.environ["REDSHIFT_WORKGROUP"]
DB = os.environ["REDSHIFT_DB_NAME"]
client = boto3.client("redshift-data", region_name=REGION)

MASK_SQL = {
    "hash": "SHA2({col}, 256)",
    "partial_last4": "REPEAT('*', GREATEST(LEN({col}) - 4, 0)) || RIGHT({col}, 4)",
    # the bare literal is rejected as ambiguously typed; cast it explicitly
    "redact": "CAST('***' AS VARCHAR)",
}


def run(sql: str, label: str, ignore: tuple = ("already exists", "already attached")) -> None:
    stmt = client.execute_statement(WorkgroupName=WORKGROUP, Database=DB, Sql=sql)
    while True:
        desc = client.describe_statement(Id=stmt["Id"])
        if desc["Status"] == "FINISHED":
            print(f"ok: {label}")
            return
        if desc["Status"] in ("FAILED", "ABORTED"):
            error = desc.get("Error", "")
            if any(phrase in error for phrase in ignore):
                print(f"skip (exists): {label}")
                return
            raise SystemExit(f"FAILED: {label}: {error}")
        time.sleep(0.5)


# 1. The read-only accounts. bi_analyst is the named service account; on
# Redshift Serverless the Data API maps the task's IAM role to an
# auto-created user "IAMR:<role-name>", and THAT is the identity the
# container's queries actually run as, so it gets the same grants and the
# same masking policies. Set TASK_ROLE to the stack's task role name.
users = ["bi_analyst"]
task_role = os.environ.get("TASK_ROLE")
if task_role:
    users.append(f"IAMR:{task_role}")

for user in users:
    quoted = f'"{user}"'
    run(f"CREATE USER {quoted} PASSWORD DISABLE", f"user {user}")
    run(f"GRANT USAGE ON SCHEMA analytics TO {quoted}", f"usage for {user}")
    run(f"GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO {quoted}", f"select for {user}")

# 2. Masking policies from the shared config.
line_re = re.compile(r"^(\w+)\.(\w+)\.(\w+):\s*(\w+)\s*$")
for line in open("config/pii_columns.yaml"):
    m = line_re.match(line.strip())
    if not m:
        continue
    schema, table, column, strategy = m.groups()
    policy = f"mask_{table}_{column}"
    expr = MASK_SQL[strategy].format(col=column)
    run(
        f"CREATE MASKING POLICY {policy} WITH ({column} VARCHAR) USING ({expr})",
        f"policy {policy}",
    )
    for user in users:
        run(
            f'ATTACH MASKING POLICY {policy} ON {schema}.{table}({column}) TO "{user}"',
            f"attach {policy} for {user}",
        )

print("provisioned: read-only account + engine-side masking")
PYEOF
