# /// script
# dependencies = ["boto3", "pandas", "pyarrow", "s3fs"]
# ///
"""Why Parquet wins: write a partitioned dataset, query it in place.

Builds a small applications table, writes it to the lake as Parquet
partitioned by vintage, and answers "average default rate by vintage" by
reading only the columns and partitions the query needs. PyArrow's dataset
reader does the pruning on the laptop; on AWS the same files are Athena's
territory.

Usage:
  uv run lake-basics/parquet_lake.py
"""

import os
import random

import boto3
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

BUCKET = os.environ.get("LAKE_BUCKET", "credit-lake")

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

# query in place: PyArrow reads only the two columns the answer needs, and the
# vintage filter prunes whole partitions before a single file is opened
if endpoint:
    host = endpoint.removeprefix("http://").removeprefix("https://")
    lake_fs = pafs.S3FileSystem(
        access_key=os.environ.get("AWS_ACCESS_KEY_ID", "local"),
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "localsecret"),
        endpoint_override=host,
        scheme="http",
    )
else:
    lake_fs = pafs.S3FileSystem()  # region + credentials from the chain

# only the Parquet files: the same prefix also holds a hand-written JSON object
# (see s3_objects.py), so read the dataset explicitly, the way Athena's table
# definition scopes itself to the curated files
parquet_files = [
    f.path
    for f in lake_fs.get_file_info(
        pafs.FileSelector(f"{BUCKET}/curated/applications", recursive=True)
    )
    if f.path.endswith(".parquet")
]
applications_ds = ds.dataset(
    parquet_files,
    filesystem=lake_fs,
    format="parquet",
    partitioning=ds.partitioning(flavor="hive"),
)
recent = applications_ds.to_table(
    columns=["vintage", "default_rate"],  # column pruning
    filter=ds.field("vintage") >= "2025-12",  # partition pruning
)
result = (
    recent.group_by("vintage")
    .aggregate([("default_rate", "mean"), ("default_rate", "count")])
    .to_pandas()
    .rename(
        columns={"default_rate_mean": "avg_dr", "default_rate_count": "applications"}
    )
    .sort_values("vintage")
)
result["avg_dr"] = result["avg_dr"].round(4)
print(result.to_string(index=False))
