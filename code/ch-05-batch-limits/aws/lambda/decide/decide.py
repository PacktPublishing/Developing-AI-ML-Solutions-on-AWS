"""Turn scores into limit decisions, as a pipeline Lambda step.

The rules are a handful of comparisons over one row at a time, so this needs neither a
model nor a dataframe: it streams scores.csv out of S3 and writes decisions.csv back,
using nothing but the standard library and boto3.

That is also why it is a Lambda rather than a Processing job. The work is bounded by
how long the rules take per customer, not by anything that needs an instance sized for
it. The step before it, which runs the model, is a different matter.
"""

import csv
import io
import os

import boto3

INCREASE_PD = float(os.environ.get("INCREASE_PD", "0.05"))
INCREASE_UTILIZATION = float(os.environ.get("INCREASE_UTILIZATION", "0.60"))
INCREASE_STEP = float(os.environ.get("INCREASE_STEP", "0.20"))
DECREASE_PD = float(os.environ.get("DECREASE_PD", "0.25"))
DECREASE_STEP = float(os.environ.get("DECREASE_STEP", "0.30"))


def _s3():
    """Build an S3 client that honours a local endpoint when one is configured."""
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT") or None)


def _split(uri: str) -> tuple[str, str]:
    """s3://bucket/key -> (bucket, key)."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def decide(row: dict) -> tuple[str, float]:
    """One customer's decision and new limit."""
    pd_12m = float(row["pd_12m"])
    utilization = float(row["utilization"])
    current = float(row["current_limit"])
    if pd_12m < INCREASE_PD and utilization > INCREASE_UTILIZATION:
        return "INCREASE", round(current * (1 + INCREASE_STEP), -1)
    if pd_12m > DECREASE_PD:
        return "DECREASE", round(current * (1 - DECREASE_STEP), -1)
    return "KEEP", current


def handler(event, _context) -> dict:
    """Read the scored book, apply the rules, write the decisions beside it."""
    scores_uri = event["scores_uri"].rstrip("/") + "/scores.csv"
    decisions_uri = event["decisions_uri"].rstrip("/") + "/decisions.csv"

    bucket, key = _split(scores_uri)
    body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    reader = csv.DictReader(io.StringIO(body))
    columns = list(reader.fieldnames or [])

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=[*columns, "decision", "new_limit"])
    writer.writeheader()
    counts: dict[str, int] = {}
    for row in reader:
        decision, new_limit = decide(row)
        counts[decision] = counts.get(decision, 0) + 1
        writer.writerow({**row, "decision": decision, "new_limit": new_limit})

    bucket, key = _split(decisions_uri)
    _s3().put_object(Bucket=bucket, Key=key, Body=out.getvalue().encode())
    print(f"decided {sum(counts.values())} customers {counts}")
    return {"decided": sum(counts.values())}
