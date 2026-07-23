# /// script
# dependencies = ["boto3"]
# ///
"""Buckets and objects: the S3 data model in a dozen calls.

Writes a scored decision as an object, reads it back, lists a prefix,
and stamps provenance metadata and tags on a curated object.

Usage:
  uv run lake-basics/s3_objects.py
"""

import json
import os

import boto3

BUCKET = os.environ.get("LAKE_BUCKET", "credit-lake")

s3 = boto3.client("s3")
try:
    s3.create_bucket(Bucket=BUCKET)
except s3.exceptions.BucketAlreadyOwnedByYou:
    pass

# write a scored decision as an object
s3.put_object(
    Bucket=BUCKET,
    Key="decisions/2026/01/app-1042.json",
    Body=json.dumps({"id": 1042, "decision": "APPROVE"}),
)

# read it back
obj = s3.get_object(Bucket=BUCKET, Key="decisions/2026/01/app-1042.json")
print("read back:", json.loads(obj["Body"].read()))

# list "a folder": the namespace is flat, a folder is a key prefix
page = s3.list_objects_v2(Bucket=BUCKET, Prefix="decisions/2026/01/")
print("keys under prefix:", [o["Key"] for o in page.get("Contents", [])])

# stamp provenance on a curated object: metadata travels with the object,
# tags can be filtered on by lifecycle, replication, and batch operations
s3.put_object(
    Bucket=BUCKET,
    Key="curated/applications/2026/01/app-1042.json",
    Body=json.dumps({"id": 1042, "vintage": "2026-01"}),
    Metadata={"source-job": "backfill-2026-01"},
    Tagging="job=backfill-2026-01&source=bureau",
)
meta = s3.head_object(Bucket=BUCKET, Key="curated/applications/2026/01/app-1042.json")
print("provenance:", meta["Metadata"]["source-job"])
