# /// script
# dependencies = ["boto3", "psycopg2-binary", "pyarrow", "pyiceberg[s3fs]"]
# ///
"""Ship the gold mart into the feature store, both the offline and online halves.

The script reads applicant_credit_profile from the warehouse and appends it
to an Iceberg table for the offline store, along with the bookkeeping columns
a SageMaker Feature Store keeps on every record (event_time, write_time,
api_invocation_time, is_deleted). It then writes the same rows, one per
applicant, into DynamoDB for the online store. Locally the Iceberg table
lives in the REST catalog over MinIO and Trino can query it; on AWS the rows
go to a feature group created with the Iceberg table format, through
feature_group.ingest().

Usage:
  uv run ingest_features.py
"""

import datetime
import os

import boto3
import psycopg2
import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError

# -------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------
WAREHOUSE_DSN = os.environ.get(
    "WAREHOUSE_DSN", "postgresql://loader:loader@localhost:5439/bureau"
)
CATALOG_URI = os.environ.get("ICEBERG_REST_URI", "http://localhost:8181")
FEATURE_BUCKET = "feature-store"
TABLE = "features.applicant_credit_profile"


# -------------------------------------------------------------------------------
# Warehouse read
# -------------------------------------------------------------------------------
def read_mart() -> pa.Table:
    """Read the gold mart into an Arrow table with offline-store columns."""
    with psycopg2.connect(WAREHOUSE_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "select applicant_id, score, state, tradeline_count,"
            " total_balance, delinquent_count"
            " from bureau_marts.applicant_credit_profile"
        )
        rows = cur.fetchall()

    if not rows:
        raise SystemExit(
            "No rows in bureau_marts.applicant_credit_profile; run dbt first"
        )

    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    columns = list(zip(*rows))
    return pa.table(
        {
            "applicant_id": pa.array(columns[0], pa.int64()),
            "score": pa.array(columns[1], pa.int64()),
            "state": pa.array(columns[2], pa.string()),
            "tradeline_count": pa.array(columns[3], pa.int64()),
            "total_balance": pa.array(columns[4], pa.int64()),
            "delinquent_count": pa.array(columns[5], pa.int64()),
            # the columns a Feature Store offline store adds to every record
            "event_time": pa.array([now] * len(rows), pa.timestamp("us")),
            "write_time": pa.array([now] * len(rows), pa.timestamp("us")),
            "api_invocation_time": pa.array([now] * len(rows), pa.timestamp("us")),
            "is_deleted": pa.array([False] * len(rows), pa.bool_()),
        }
    )


# -------------------------------------------------------------------------------
# Offline append and entrypoint
# -------------------------------------------------------------------------------
def main() -> None:
    """Append the mart to the Iceberg feature table and the online store."""
    features = read_mart()

    s3 = boto3.client("s3")
    try:
        s3.create_bucket(Bucket=FEATURE_BUCKET)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass

    catalog = load_catalog(
        "local",
        uri=CATALOG_URI,
        warehouse=f"s3://{FEATURE_BUCKET}/",
        **{
            "s3.endpoint": os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000"),
            "s3.access-key-id": os.environ.get("AWS_ACCESS_KEY_ID", "local"),
            "s3.secret-access-key": os.environ.get(
                "AWS_SECRET_ACCESS_KEY", "localsecret"
            ),
            "s3.region": "us-east-1",
        },
    )
    try:
        catalog.create_namespace("features")
    except NamespaceAlreadyExistsError:
        pass
    try:
        table = catalog.load_table(TABLE)
    except NoSuchTableError:
        table = catalog.create_table(TABLE, schema=features.schema)

    table.append(features)
    print(f"Appended {features.num_rows} feature rows to {TABLE}")

    write_online(features)


# -------------------------------------------------------------------------------
# Online store
# -------------------------------------------------------------------------------
def write_online(features: pa.Table) -> None:
    """Write the latest feature row per applicant to DynamoDB."""
    # the online-store half: latest row per applicant into DynamoDB for
    # millisecond lookups at serving time
    ddb = boto3.resource(
        "dynamodb",
        endpoint_url=os.environ.get("ONLINE_STORE_URL", "http://localhost:8000"),
    )
    try:
        ddb.create_table(
            TableName="applicant-credit-profile",
            KeySchema=[{"AttributeName": "applicant_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "applicant_id", "AttributeType": "N"}
            ],
            BillingMode="PAY_PER_REQUEST",
        ).wait_until_exists()
    except ddb.meta.client.exceptions.ResourceInUseException:
        pass

    table = ddb.Table("applicant-credit-profile")
    rows = features.to_pylist()
    with table.batch_writer(overwrite_by_pkeys=["applicant_id"]) as batch:
        for row in rows:
            batch.put_item(
                Item={
                    "applicant_id": row["applicant_id"],
                    "score": row["score"],
                    "state": row["state"],
                    "tradeline_count": row["tradeline_count"],
                    "total_balance": row["total_balance"],
                    "delinquent_count": row["delinquent_count"],
                    "event_time": row["event_time"].isoformat(),
                }
            )
    print(f"Wrote {len(rows)} latest feature rows to the online store")


if __name__ == "__main__":
    main()
