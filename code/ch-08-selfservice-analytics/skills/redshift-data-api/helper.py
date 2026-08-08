"""Redshift Data API helper: read-only, bounded, masked.

Runs inside the image's project environment (/app, boto3 in pyproject.toml); no inline script metadata, which would make uv build a separate environment per call.

Usage:
    from helper import execute_redshift_query
    rows = execute_redshift_query("SELECT ...")   # list[dict]

Env vars:
    REDSHIFT_REGION, REDSHIFT_WORKGROUP (serverless) or REDSHIFT_HOST (shim),
    REDSHIFT_DB_NAME, REDSHIFT_USER_NAME
    REDSHIFT_QUERY_TIMEOUT=30    seconds before the statement is cancelled
    REDSHIFT_MAX_ROWS=10000      result row cap

The same code runs against real AWS and the local shim: boto3 picks up AWS_ENDPOINT_URL_REDSHIFT_DATA when set, pointing this client at the from-source Data API in front of redshift-local.
Verified against the local stack 2026-07-30 (helper through the shim, masking, and a full Claude Code run); the cloud path is still unrun.
"""

import json
import os
import re
import sys
import time
from typing import Any

import boto3
from masking import mask_records

_TIMEOUT = int(os.environ.get("REDSHIFT_QUERY_TIMEOUT", "30"))
_MAX_ROWS = int(os.environ.get("REDSHIFT_MAX_ROWS", "10000"))

_WRITE_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|TRUNCATE|ALTER|GRANT|REVOKE|COPY|UNLOAD|CALL|MERGE)\b",
    re.IGNORECASE,
)


def _audit(event: str, **fields: Any) -> None:
    """One structured line per event, written to the container's main stdout (/proc/1/fd/1, which awslogs ships to CloudWatch; falls back to stderr outside a container so the terminal stream stays clean)."""
    line = json.dumps({"audit": event, **fields}, default=str)
    try:
        with open("/proc/1/fd/1", "w") as out:
            out.write(line + "\n")
    except OSError:
        print(line, file=sys.stderr, flush=True)


def execute_redshift_query(sql: str) -> list[dict]:
    """Run one read-only statement and return rows as dicts."""
    if _WRITE_RE.match(sql):
        raise ValueError("read-only helper: statement rejected")

    client = boto3.client("redshift-data", region_name=os.environ["REDSHIFT_REGION"])
    params: dict[str, Any] = {
        "Database": os.environ["REDSHIFT_DB_NAME"],
        "Sql": sql,
    }
    workgroup = os.environ.get("REDSHIFT_WORKGROUP")
    if workgroup:
        params["WorkgroupName"] = workgroup  # serverless: IAM auth, no db user
    else:
        params["ClusterIdentifier"] = os.environ.get("REDSHIFT_HOST", "local")
        params["DbUser"] = os.environ.get("REDSHIFT_USER_NAME", "bi_analyst")

    _audit("query_start", sql=sql, user=os.environ.get("REDSHIFT_USER_NAME"))
    stmt = client.execute_statement(**params)
    sid = stmt["Id"]

    deadline = time.monotonic() + _TIMEOUT
    while True:
        desc = client.describe_statement(Id=sid)
        status = desc["Status"]
        if status == "FINISHED":
            break
        if status in ("FAILED", "ABORTED"):
            _audit("query_failed", id=sid, error=desc.get("Error"))
            raise RuntimeError(desc.get("Error", status))
        if time.monotonic() > deadline:
            client.cancel_statement(Id=sid)
            _audit("query_timeout", id=sid, timeout_s=_TIMEOUT)
            raise TimeoutError(f"query exceeded {_TIMEOUT}s and was cancelled")
        time.sleep(0.5)

    if not desc.get("HasResultSet"):
        return []

    rows: list[dict] = []
    token = None
    columns = None
    while True:
        kwargs = {"Id": sid}
        if token:
            kwargs["NextToken"] = token
        page = client.get_statement_result(**kwargs)
        if columns is None:
            columns = [c["name"] for c in page["ColumnMetadata"]]
        for rec in page["Records"]:
            row = {}
            for col, field in zip(columns, rec):
                row[col] = None if field.get("isNull") else next(iter(field.values()))
            rows.append(row)
            if len(rows) >= _MAX_ROWS:
                _audit("row_cap", id=sid, cap=_MAX_ROWS)
                return mask_records(rows)
        token = page.get("NextToken")
        if not token:
            break

    _audit("query_done", id=sid, rows=len(rows))
    return mask_records(rows)


if __name__ == "__main__":
    # CLI mode: run one query and print each row as a JSON line. This is the
    # path the assistant uses, so it never writes row-handling code of its own.
    if len(sys.argv) != 2:
        sys.exit("usage: uv run helper.py 'SELECT ...'")
    for record in execute_redshift_query(sys.argv[1]):
        print(json.dumps(record, default=str))
