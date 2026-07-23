# /// script
# dependencies = ["boto3", "pandas", "pyarrow", "duckdb", "s3fs"]
# ///
"""Why Parquet wins: write a partitioned dataset, query it in place.

Builds a small applications table, writes it to the lake as Parquet
partitioned by vintage, and answers "average PD by vintage" by scanning
only the columns and partitions the query needs. DuckDB stands in for
the laptop-scale reader; on AWS the same files are Athena's territory.

Usage:
  uv run lake-basics/parquet_lake.py
"""

import os
import random

import boto3
import duckdb
import pandas as pd

BUCKET = os.environ.get("LAKE_BUCKET", "credit-lake")

random.seed("parquet-lake")
applications = pd.DataFrame(
    {
        "application_id": range(1000, 1600),
        "vintage": [
            random.choice(["2025-11", "2025-12", "2026-01"]) for _ in range(600)
        ],
        "pd": [round(random.uniform(0.01, 0.25), 4) for _ in range(600)],
        "state": [random.choice(["CA", "TX", "NY"]) for _ in range(600)],
    }
)

s3 = boto3.client("s3")
try:
    s3.create_bucket(Bucket=BUCKET)
except s3.exceptions.BucketAlreadyOwnedByYou:
    pass

# endpoint set -> the local cloud; unset -> real AWS via the credential chain
endpoint = os.environ.get("AWS_ENDPOINT_URL")
storage_options = {"client_kwargs": {"endpoint_url": endpoint}} if endpoint else None
applications.to_parquet(
    f"s3://{BUCKET}/curated/applications/",
    partition_cols=["vintage"],
    storage_options=storage_options,
)
print("wrote partitioned Parquet to s3://credit-lake/curated/applications/")

# query in place: only two columns are read, and the vintage filter
# prunes whole partitions before a single file is opened
duckdb.sql("install httpfs; load httpfs;")
if endpoint:
    host = endpoint.removeprefix("http://").removeprefix("https://")
    duckdb.sql(f"""
        set s3_endpoint='{host}'; set s3_use_ssl=false; set s3_url_style='path';
        set s3_access_key_id='{os.environ.get("AWS_ACCESS_KEY_ID", "local")}';
        set s3_secret_access_key='{os.environ.get("AWS_SECRET_ACCESS_KEY", "localsecret")}';
    """)
else:
    duckdb.sql("create or replace secret aws (type s3, provider credential_chain);")
result = duckdb.sql(f"""
    select vintage, round(avg(pd), 4) as mean_pd, count(*) as applications
    from read_parquet('s3://{BUCKET}/curated/applications/*/*.parquet', hive_partitioning=true)
    where vintage >= '2025-12'
    group by vintage order by vintage
""")
print(result)
