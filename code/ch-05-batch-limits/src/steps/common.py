"""Shared plumbing for the batch steps: the warehouse DSN and the S3 client.

Locally the endpoints point at the compose containers; on AWS the same
variables point at the real services, and this module does not change.
"""

import os

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
WAREHOUSE_DSN = os.environ.get(
    "WAREHOUSE_DSN", "postgresql://analyst:analyst@localhost:5439/portfolio"
)
BUCKET = os.environ.get("BATCH_BUCKET", "portfolio")
PREFIX = "batch"


def s3_client():
    """Build an S3 client for the local container or the real service."""
    return boto3.client("s3", region_name=REGION, endpoint_url=S3_ENDPOINT or None)
