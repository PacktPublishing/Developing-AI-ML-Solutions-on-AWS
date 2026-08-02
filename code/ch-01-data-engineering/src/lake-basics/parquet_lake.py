# /// script
# dependencies = ["awswrangler", "pandas"]
# ///
"""Why Parquet wins: write a partitioned dataset, query it in place.

Builds a small applications table, writes it to the lake as Parquet
partitioned by vintage, and answers "average default rate by vintage" by
reading only the columns and partitions the query needs. awswrangler (the AWS
SDK for pandas) does the pruning; the same calls run against the local S3
stand-in or real S3 — only the endpoint changes.

Usage:
  uv run lake-basics/parquet_lake.py
"""

import os
import random

import awswrangler as wr
import boto3
import pandas as pd

BUCKET = os.environ.get("LAKE_BUCKET", "credit-lake")

# One library, one code path, local or AWS. awswrangler resolves credentials
# through the boto3 session — env vars, a named profile, or an assumed role such
# as ch01-user — and talks to whichever S3 endpoint is set: the local S3 stand-in
# when AWS_ENDPOINT_URL points at it, real S3 otherwise.
endpoint = os.environ.get("AWS_ENDPOINT_URL")
if endpoint:
    wr.config.s3_endpoint_url = endpoint
session = boto3.Session()

random.seed("parquet-lake")
applications = pd.DataFrame(
    {
        "application_id": range(1000, 1600),
        "vintage": [
            random.choice(["2025-11", "2025-12", "2026-01"]) for _ in range(600)
        ],
        "default_rate": [round(random.uniform(0.01, 0.25), 4) for _ in range(600)],
        "state": [random.choice(["CA", "TX", "NY"]) for _ in range(600)],
    }
)

# create the lake bucket if this is a fresh stack (idempotent)
s3 = session.client("s3", endpoint_url=endpoint)
try:
    s3.create_bucket(Bucket=BUCKET)
except s3.exceptions.BucketAlreadyOwnedByYou:
    pass

wr.s3.to_parquet(
    df=applications,
    path=f"s3://{BUCKET}/curated/applications/",
    dataset=True,
    partition_cols=["vintage"],
    boto3_session=session,
)
print(f"wrote partitioned Parquet to s3://{BUCKET}/curated/applications/")

# query in place: read only the two columns the answer needs, and let the vintage
# filter prune whole partitions before a single file is opened
recent = wr.s3.read_parquet(
    path=f"s3://{BUCKET}/curated/applications/",
    dataset=True,
    columns=["vintage", "default_rate"],  # column pruning
    partition_filter=lambda part: part["vintage"] >= "2025-12",  # partition pruning
    boto3_session=session,
)
result = (
    recent.groupby("vintage")["default_rate"]
    .agg(avg_dr="mean", applications="count")
    .reset_index()
    .sort_values("vintage")
)
result["avg_dr"] = result["avg_dr"].round(4)
print(result.to_string(index=False))
