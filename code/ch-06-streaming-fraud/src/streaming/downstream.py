# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "psycopg2-binary"]
# ///
"""Create the analytics table and the delivery stream that feeds it.

Two setup steps, in the order Firehose requires them. First the destination
table: Firehose loads into an existing table and never creates one, on AWS
and against the local service alike, so the table is made here, the way the
analytics owner would. Then the delivery stream: the one boto3 call the
local world needs that CloudFormation makes on AWS — CreateDeliveryStream
with a Kinesis source and a Redshift destination. The configuration dict
below is the same shape, field for field, as the
RedshiftDestinationConfiguration in aws/template.yaml, which is the point:
one destination config, declared in YAML for the cloud and passed to the
API locally, delivered by the firehose-local service against kinesalite,
S3Proxy, and the local warehouse.

Idempotent: table and delivery stream are both left alone if they exist.

Usage (with `uv run firehose-local/app.py` serving in another terminal):
  uv run streaming/downstream.py
"""

import os

import psycopg2
from common import (
    DECISIONS_STREAM,
    REGION,
    ensure_stream,
    firehose_client,
    kinesis_client,
)

DELIVERY_STREAM = "decisions-to-warehouse"
STAGING_BUCKET = "fraud-analytics"
LOCAL_ROLE = "arn:aws:iam::000000000000:role/firehose-local"
WAREHOUSE_JDBC = os.environ.get(
    "WAREHOUSE_JDBC", "jdbc:redshift://localhost:5439/fraud"
)
WAREHOUSE_USER = os.environ.get("WAREHOUSE_USER", "analyst")
WAREHOUSE_PASSWORD = os.environ.get("WAREHOUSE_PASSWORD", "analyst")

DDL = """
CREATE TABLE IF NOT EXISTS fraud_decisions (
    transaction_id VARCHAR(32),
    user_id        VARCHAR(16),
    merchant_id    VARCHAR(16),
    amount_usd     NUMERIC(12, 2),
    score          DOUBLE PRECISION,
    decision       VARCHAR(8),
    event_time     TIMESTAMP,
    latency_ms     DOUBLE PRECISION
)
"""


def ensure_table() -> None:
    """Create the destination table the way the analytics owner would."""
    hostport_db = WAREHOUSE_JDBC.removeprefix("jdbc:redshift://")
    hostport, _, db = hostport_db.partition("/")
    dsn = f"postgresql://{WAREHOUSE_USER}:{WAREHOUSE_PASSWORD}@{hostport}/{db}"
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(DDL)


def main() -> None:
    """Create the table, then the delivery stream, unless they exist."""
    ensure_table()
    ensure_stream(kinesis_client(), DECISIONS_STREAM)
    firehose = firehose_client()
    existing = firehose.list_delivery_streams()["DeliveryStreamNames"]
    if DELIVERY_STREAM in existing:
        print(f"{DELIVERY_STREAM} already exists")
        return

    # The same destination configuration aws/template.yaml declares — the
    # local call and the CloudFormation resource read identically.
    firehose.create_delivery_stream(
        DeliveryStreamName=DELIVERY_STREAM,
        DeliveryStreamType="KinesisStreamAsSource",
        KinesisStreamSourceConfiguration={
            "KinesisStreamARN": f"arn:aws:kinesis:{REGION}:000000000000:stream/{DECISIONS_STREAM}",
            "RoleARN": LOCAL_ROLE,
        },
        RedshiftDestinationConfiguration={
            "ClusterJDBCURL": WAREHOUSE_JDBC,
            "Username": WAREHOUSE_USER,
            "Password": WAREHOUSE_PASSWORD,
            "RoleARN": LOCAL_ROLE,
            "CopyCommand": {
                "DataTableName": "fraud_decisions",
                "CopyOptions": "FORMAT AS JSON 'auto'",
            },
            "S3Configuration": {
                "BucketARN": f"arn:aws:s3:::{STAGING_BUCKET}",
                "Prefix": "decisions/",
                "RoleARN": LOCAL_ROLE,
                "BufferingHints": {"IntervalInSeconds": 10, "SizeInMBs": 1},
                "CompressionFormat": "UNCOMPRESSED",
            },
        },
    )
    status = firehose.describe_delivery_stream(DeliveryStreamName=DELIVERY_STREAM)
    print(
        f"{DELIVERY_STREAM}: "
        f"{status['DeliveryStreamDescription']['DeliveryStreamStatus']}"
    )


if __name__ == "__main__":
    main()
