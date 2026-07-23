# /// script
# dependencies = ["boto3", "dlt[postgres,redshift]"]
# ///
"""Load one day of bureau files from the raw zone into the warehouse.

It runs as a Glue Python shell job, so it is plain Python with no Spark. The
job lists the day's files, reads the nested JSON, and hands the records to
dlt, which flattens them into parent and child tables and widens the schema
on its own whenever the bureau adds a field. dlt's redshift destination does
the loading in both worlds: locally it points at redshift-local, on AWS at a
real Redshift cluster. The code is the same either way; only the credentials
change.

Usage:
  uv run glue_bureau_job.py --date 2026-07-17
"""

import argparse
import datetime
import json
import os

import boto3
import dlt

BUCKET = os.environ.get("RAW_BUCKET", "bureau-raw")
DESTINATION = os.environ.get("DLT_DESTINATION", "redshift")


def read_day(s3, bucket: str, day: str) -> list[dict]:
    """Read one day of raw bureau JSON records from S3."""
    prefix = f"raw/dt={day}/"
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    if not keys:
        raise SystemExit(f"No files under s3://{bucket}/{prefix}")

    records = []
    for key in keys:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        records.extend(json.loads(line) for line in body.splitlines() if line)
    print(f"Read {len(records)} records from {len(keys)} files under {prefix}")
    return records


def main() -> None:
    """Run the load: read the day's records and hand them to dlt."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", default=str(datetime.date.today() - datetime.timedelta(days=1))
    )
    # destination overridable per environment: redshift when a warehouse is
    # reachable, filesystem to land dlt's output in S3 as a bronze dataset
    parser.add_argument("--destination", default=DESTINATION)
    parser.add_argument("--raw-bucket", default=BUCKET)
    parser.add_argument("--bucket-url", default=None)
    # parse_known_args so Glue's own job arguments (--job-name, ...) pass through
    args, _ = parser.parse_known_args()

    if args.bucket_url:
        os.environ["DESTINATION__FILESYSTEM__BUCKET_URL"] = args.bucket_url
    # dlt state must live under /tmp on Glue and Lambda (read-only filesystem)
    os.environ.setdefault("DLT_DATA_DIR", "/tmp/dlt")

    records = read_day(boto3.client("s3"), args.raw_bucket, args.date)

    pipeline = dlt.pipeline(
        pipeline_name="bureau",
        destination=args.destination,
        dataset_name="bureau_raw",
    )
    info = pipeline.run(records, table_name="reports", write_disposition="append")
    print(f"Loaded to {args.destination}: {info.dataset_name}.reports")


if __name__ == "__main__":
    main()
