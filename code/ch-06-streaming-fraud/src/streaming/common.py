"""Shared plumbing for the chapter's streaming scripts.

One place for the boto3 clients, the stream names, and the two helpers every
stage needs: create-a-stream-if-missing and iterate-records-across-shards.
Locally the endpoints point at the containers from docker-compose; on AWS the
same variables point at the real services, and this module does not change.
"""

import json
import os
import time
from collections.abc import Iterator

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
KINESIS_ENDPOINT = os.environ.get("KINESIS_ENDPOINT", "http://localhost:4567")
DYNAMODB_ENDPOINT = os.environ.get("DYNAMODB_ENDPOINT", "http://localhost:8000")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "http://localhost:8080")
FIREHOSE_ENDPOINT = os.environ.get("FIREHOSE_ENDPOINT", "http://localhost:8008")

TRANSACTIONS_STREAM = "transactions"
DECISIONS_STREAM = "decisions"
BLOCKS_TABLE = "blocked_transactions"
SCORING_ENDPOINT_NAME = "fraud-scoring"


def kinesis_client():
    """Build a Kinesis client for the local container or the real service."""
    return boto3.client(
        "kinesis", region_name=REGION, endpoint_url=KINESIS_ENDPOINT or None
    )


def dynamodb_resource():
    """Build a DynamoDB resource for the local container or the real service."""
    return boto3.resource(
        "dynamodb", region_name=REGION, endpoint_url=DYNAMODB_ENDPOINT or None
    )


def s3_client():
    """Build an S3 client for the local container or the real service."""
    return boto3.client("s3", region_name=REGION, endpoint_url=S3_ENDPOINT or None)


def sagemaker_runtime_client():
    """Build a SageMaker runtime client for the local serving container or AWS.

    invoke_endpoint posts to /endpoints/<name>/invocations; the local serving
    container accepts that path, so the same call reaches either world.
    """
    return boto3.client(
        "sagemaker-runtime", region_name=REGION, endpoint_url=SAGEMAKER_ENDPOINT or None
    )


def firehose_client():
    """Build a Firehose client for the local shim or the real service."""
    return boto3.client(
        "firehose", region_name=REGION, endpoint_url=FIREHOSE_ENDPOINT or None
    )


def ensure_stream(kinesis, name: str, shards: int = 2) -> None:
    """Create the stream if it does not exist and wait until it is ACTIVE."""
    try:
        kinesis.create_stream(StreamName=name, ShardCount=shards)
    except kinesis.exceptions.ResourceInUseException:
        pass
    while True:
        summary = kinesis.describe_stream_summary(StreamName=name)
        if summary["StreamDescriptionSummary"]["StreamStatus"] == "ACTIVE":
            return
        time.sleep(0.5)


def iter_records(
    kinesis, stream: str, from_start: bool = True, idle_seconds: float = 5.0
) -> Iterator[dict]:
    """Yield JSON records from every shard until the stream goes quiet.

    Polls all shards round-robin and exits once no shard has produced a record
    for idle_seconds. This is the poller half of what Lambda's Kinesis event
    source mapping does for you on AWS: track a per-shard iterator, fetch
    batches, hand records downstream.
    """
    shards = kinesis.list_shards(StreamName=stream)["Shards"]
    iterator_type = "TRIM_HORIZON" if from_start else "LATEST"
    iterators = {
        s["ShardId"]: kinesis.get_shard_iterator(
            StreamName=stream, ShardId=s["ShardId"], ShardIteratorType=iterator_type
        )["ShardIterator"]
        for s in shards
    }
    last_record = time.monotonic()
    while time.monotonic() - last_record < idle_seconds:
        for shard_id, iterator in list(iterators.items()):
            out = kinesis.get_records(ShardIterator=iterator, Limit=500)
            iterators[shard_id] = out["NextShardIterator"]
            for record in out["Records"]:
                last_record = time.monotonic()
                yield json.loads(record["Data"])
        time.sleep(0.2)
