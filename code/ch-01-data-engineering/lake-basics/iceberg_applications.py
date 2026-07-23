# /// script
# dependencies = ["boto3", "pyarrow", "pyiceberg[s3fs]"]
# ///
"""A better table: applications as Iceberg, appended and read back.

Creates an Iceberg applications table through the REST catalog (the same
open format S3 Tables manages for you on AWS), appends rows, and reads
them back. Query it with Trino too: make lake-table-query.

Usage:
  uv run lake-basics/iceberg_applications.py
"""

import os
import random

import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError

TABLE = "lake.applications"

random.seed("iceberg-lake")
rows = pa.table(
    {
        "application_id": pa.array(range(2000, 2200), pa.int64()),
        "vintage": pa.array(
            [random.choice(["2025-12", "2026-01"]) for _ in range(200)], pa.string()
        ),
        "pd": pa.array([round(random.uniform(0.01, 0.25), 4) for _ in range(200)]),
        "decision": pa.array(
            [random.choice(["APPROVE", "REFER", "DECLINE"]) for _ in range(200)],
            pa.string(),
        ),
    }
)

# S3TABLES_ARN set -> real S3 Tables through its Iceberg REST endpoint;
# unset -> the local REST catalog over MinIO. Same open format either way.
if arn := os.environ.get("S3TABLES_ARN"):
    region = arn.split(":")[3]
    catalog = load_catalog(
        "s3tables",
        **{
            "type": "rest",
            "uri": f"https://s3tables.{region}.amazonaws.com/iceberg",
            "warehouse": arn,
            "rest.sigv4-enabled": "true",
            "rest.signing-name": "s3tables",
            "rest.signing-region": region,
        },
    )
else:
    catalog = load_catalog(
        "local",
        uri=os.environ.get("ICEBERG_REST_URI", "http://localhost:8181"),
        warehouse="s3://feature-store/",
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
    catalog.create_namespace("lake")
except NamespaceAlreadyExistsError:
    pass
try:
    table = catalog.load_table(TABLE)
except NoSuchTableError:
    table = catalog.create_table(TABLE, schema=rows.schema)

table.append(rows)
scanned = table.scan().to_arrow()
print(f"appended {rows.num_rows} rows; table now holds {scanned.num_rows}")
print(scanned.slice(0, 3).to_pydict())
