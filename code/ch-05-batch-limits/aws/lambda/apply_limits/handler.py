"""Apply the staged limit decisions to the warehouse, as a pipeline Lambda step.

The decide step writes a wide decisions.csv (every scored column plus decision and
new_limit). This stages the customers whose limit actually moves into a temp table and
applies them with one set-based UPDATE ... FROM, which is what Redshift wants: a single
join rather than a statement per customer.

It talks to the warehouse the same way every other step does, over WAREHOUSE_DSN, so
the identical handler runs against redshift-local on a laptop and against Redshift
Serverless in the cloud. On AWS that means the function joins the warehouse's VPC.

Applying the limits is only half the job. The run also has to say what it did, so the
same function publishes the population counts: an SNS message a person reads, and a
CloudWatch metric a dashboard or an alarm can watch. Locally SNS_ENDPOINT points at the
chapter's events shim and the metric is skipped, there being no local CloudWatch.
"""

import csv
import io
import os

import boto3
import psycopg2
from psycopg2.extras import execute_values

TOPIC_ARN = os.environ.get("TOPIC_ARN", "")
SNS_ENDPOINT = os.environ.get("SNS_ENDPOINT", "")
METRIC_NAMESPACE = os.environ.get("METRIC_NAMESPACE", "")


def _s3():
    """Build an S3 client that honours a local endpoint when one is configured."""
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT") or None)


def _split(uri: str) -> tuple[str, str]:
    """s3://bucket/key -> (bucket, key)."""
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


def _read(decisions_uri: str) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Return the rows whose limit moves, and the count of every decision."""
    bucket, key = _split(decisions_uri)
    body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read().decode()

    changes: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for row in csv.DictReader(io.StringIO(body)):
        decision = row["decision"]
        counts[decision] = counts.get(decision, 0) + 1
        if decision != "KEEP":
            changes.append((row["customer_id"], row["new_limit"]))
    return changes, counts


def _apply(changes: list[tuple[str, str]]) -> None:
    """Stage the changes and apply them in one UPDATE ... FROM."""
    with psycopg2.connect(os.environ["WAREHOUSE_DSN"]) as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE changes "
            "(customer_id VARCHAR(16), new_limit NUMERIC(12, 2))"
        )
        execute_values(
            cur, "INSERT INTO changes (customer_id, new_limit) VALUES %s", changes
        )
        cur.execute(
            "UPDATE customers SET current_limit = c.new_limit "
            "FROM changes c WHERE customers.customer_id = c.customer_id"
        )


def _publish(counts: dict[str, int], applied: int) -> None:
    """Tell a person, and tell a dashboard, what the run did."""
    detail = ", ".join(
        f"{name.lower()} {count}" for name, count in sorted(counts.items())
    )
    if TOPIC_ARN:
        boto3.client("sns", endpoint_url=SNS_ENDPOINT or None).publish(
            TopicArn=TOPIC_ARN,
            Subject="batch limit run complete",
            Message=f"applied {applied} limit changes ({detail})",
        )
    if METRIC_NAMESPACE:
        boto3.client("cloudwatch").put_metric_data(
            Namespace=METRIC_NAMESPACE,
            MetricData=[
                {"MetricName": name.title(), "Value": float(count), "Unit": "Count"}
                for name, count in counts.items()
            ],
        )


def handler(event, _context) -> dict:
    """Apply the run's limit changes, then publish what the run did."""
    changes, counts = _read(event["decisions_uri"].rstrip("/") + "/decisions.csv")
    if changes:
        _apply(changes)
    _publish(counts, len(changes))
    print(f"applied {len(changes)} limit changes {counts}")
    return {"applied": len(changes)}
