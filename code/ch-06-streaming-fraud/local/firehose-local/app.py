# /// script
# requires-python = ">=3.12"
# dependencies = ["starlette", "uvicorn", "boto3", "psycopg2-binary"]
# ///
"""A local Amazon Data Firehose: delivery streams as a generic service.

Serves the Firehose control plane (CreateDeliveryStream, DescribeDeliveryStream,
ListDeliveryStreams, DeleteDeliveryStream, and PutRecord/PutRecordBatch for
DirectPut) and runs one delivery loop per stream. Every choice comes from the
stream's own configuration: a Kinesis or DirectPut source, an S3 or Redshift
destination, records buffered until the configured size or interval trips and
staged as one object, then loaded into a Redshift destination with a single
COPY. Firehose never creates the destination table; that is the owner's job.

One documented divergence: real Redshift COPYs staged JSON directly (FORMAT AS
JSON 'auto'); the local warehouse is Postgres, whose COPY has no JSON mode, so
the loader reshapes each staged batch to CSV before the COPY.

Usage:
  uv run firehose-local/app.py          # serves on :8008
"""

import base64
import csv
import io
import json
import os
import threading
import time
import uuid
from datetime import UTC, datetime

import boto3
import psycopg2
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
KINESIS_ENDPOINT = os.environ.get("KINESIS_ENDPOINT", "http://localhost:4567")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
PORT = int(os.environ.get("FIREHOSE_PORT", "8008"))

# name -> {"config", "buffer" (DirectPut), "stop", "thread", "created"}
streams: dict[str, dict] = {}


def stream_arn(name: str) -> str:
    """Build the delivery stream's ARN."""
    return f"arn:aws:firehose:{REGION}:000000000000:deliverystream/{name}"


def jdbc_to_dsn(jdbc_url: str, username: str, password: str) -> str:
    """Turn Firehose's ClusterJDBCURL into a psycopg2 DSN."""
    hostport_db = jdbc_url.removeprefix("jdbc:redshift://")
    hostport, _, db = hostport_db.partition("/")
    return f"postgresql://{username}:{password}@{hostport}/{db}"


def object_key(prefix: str, stream_name: str, now: datetime) -> str:
    """Build the Firehose key layout: prefix/YYYY/MM/DD/HH/name-timestamp-uuid."""
    return (
        f"{prefix}{now:%Y/%m/%d/%H}/"
        f"{stream_name}-{now:%Y-%m-%d-%H-%M-%S}-{uuid.uuid4()}"
    )


def iter_json_records(staged: bytes):
    """Yield the JSON objects in a staged batch.

    Firehose concatenates records as the producer sent them, with or without
    newlines, so the parser walks the byte stream object by object rather than
    assuming one record per line.
    """
    decoder = json.JSONDecoder()
    text = staged.decode()
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        record, pos = decoder.raw_decode(text, pos)
        yield record


def load_batch(conn, table: str, staged: bytes) -> None:
    """COPY one staged batch into the destination table.

    Real Redshift runs `COPY ... FORMAT AS JSON 'auto'` straight off the
    staged object, matching JSON keys to table columns. Postgres COPY has no
    JSON mode, so this loader asks the catalog for the table's column order and
    reshapes the same staged bytes to CSV, the mapping semantics of JSON 'auto'
    over the transport Postgres understands.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s ORDER BY ordinal_position",
            (table,),
        )
        columns = [row[0] for row in cur.fetchall()]
    if not columns:
        raise RuntimeError(f"destination table {table} does not exist")

    body = io.StringIO()
    writer = csv.DictWriter(body, fieldnames=columns, extrasaction="ignore")
    for record in iter_json_records(staged):
        writer.writerow({c: record.get(c) for c in columns})
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} FROM STDIN WITH (FORMAT csv)",
            io.StringIO(body.getvalue()),
        )
    conn.commit()


def deliver(name: str, stream: dict) -> None:
    """Run the delivery loop for one stream: consume, buffer, stage, load."""
    config = stream["config"]
    stop = stream["stop"]

    # Destination: Redshift wraps an S3 staging config; plain S3 stands alone.
    redshift = config.get("RedshiftDestinationConfiguration")
    s3_conf = (
        redshift["S3Configuration"]
        if redshift
        else config.get("ExtendedS3DestinationConfiguration")
        or config["S3DestinationConfiguration"]
    )
    bucket = s3_conf["BucketARN"].rpartition(":::")[2]
    prefix = s3_conf.get("Prefix", "")
    hints = s3_conf.get("BufferingHints", {})
    max_bytes = hints.get("SizeInMBs", 5) * 1024 * 1024
    max_seconds = hints.get("IntervalInSeconds", 300)

    s3 = boto3.client("s3", region_name=REGION, endpoint_url=S3_ENDPOINT)
    conn = None
    if redshift:
        conn = psycopg2.connect(
            jdbc_to_dsn(
                redshift["ClusterJDBCURL"], redshift["Username"], redshift["Password"]
            )
        )

    # Source: shard iterators for a Kinesis source, the shared in-memory
    # buffer for DirectPut.
    iterators = {}
    if "KinesisStreamSourceConfiguration" in config:
        source_arn = config["KinesisStreamSourceConfiguration"]["KinesisStreamARN"]
        source_name = source_arn.rpartition("/")[2]
        kinesis = boto3.client(
            "kinesis", region_name=REGION, endpoint_url=KINESIS_ENDPOINT
        )
        iterators = {
            shard["ShardId"]: kinesis.get_shard_iterator(
                StreamName=source_name,
                ShardId=shard["ShardId"],
                ShardIteratorType="TRIM_HORIZON",
            )["ShardIterator"]
            for shard in kinesis.list_shards(StreamName=source_name)["Shards"]
        }

    batch: list[bytes] = []
    batched = 0
    deadline = time.monotonic() + max_seconds
    while not stop.is_set():
        for shard_id, iterator in list(iterators.items()):
            out = kinesis.get_records(ShardIterator=iterator, Limit=500)
            iterators[shard_id] = out["NextShardIterator"]
            for record in out["Records"]:
                batch.append(bytes(record["Data"]))
                batched += len(record["Data"])
        while stream["buffer"]:
            data = stream["buffer"].pop(0)
            batch.append(data)
            batched += len(data)
        if batch and (batched >= max_bytes or time.monotonic() >= deadline):
            staged = b"\n".join(batch)
            key = object_key(prefix, name, datetime.now(UTC))
            s3.put_object(Bucket=bucket, Key=key, Body=staged)
            if redshift:
                load_batch(conn, redshift["CopyCommand"]["DataTableName"], staged)
            print(
                f"[{name}] delivered {len(batch)} records via s3://{bucket}/{key}",
                flush=True,
            )
            batch, batched = [], 0
            deadline = time.monotonic() + max_seconds
        stop.wait(0.2)
    if conn is not None:
        conn.close()


def describe(name: str) -> dict:
    """Build the DescribeDeliveryStream response body for one stream."""
    stream = streams[name]
    config = stream["config"]
    stream_type = config.get("DeliveryStreamType", "DirectPut")
    return {
        "DeliveryStreamDescription": {
            "DeliveryStreamName": name,
            "DeliveryStreamARN": stream_arn(name),
            "DeliveryStreamStatus": "ACTIVE",
            "DeliveryStreamType": stream_type,
            "VersionId": "1",
            "CreateTimestamp": stream["created"],
            "Destinations": [{"DestinationId": "destinationId-000000000001"}],
            "HasMoreDestinations": False,
        }
    }


async def dispatch(request: Request) -> JSONResponse:
    """Route one AWS-json call by its X-Amz-Target header."""
    operation = request.headers.get("x-amz-target", "").rpartition(".")[2]
    body = json.loads(await request.body() or b"{}")
    name = body.get("DeliveryStreamName", "")

    if operation == "CreateDeliveryStream":
        stream = {
            "config": body,
            "buffer": [],
            "stop": threading.Event(),
            "created": time.time(),
        }
        stream["thread"] = threading.Thread(
            target=deliver, args=(name, stream), daemon=True
        )
        streams[name] = stream
        stream["thread"].start()
        return JSONResponse({"DeliveryStreamARN": stream_arn(name)})
    if operation == "DescribeDeliveryStream" and name in streams:
        return JSONResponse(describe(name))
    if operation == "ListDeliveryStreams":
        return JSONResponse(
            {"DeliveryStreamNames": sorted(streams), "HasMoreDeliveryStreams": False}
        )
    if operation == "DeleteDeliveryStream" and name in streams:
        streams.pop(name)["stop"].set()
        return JSONResponse({})
    if operation == "PutRecord" and name in streams:
        streams[name]["buffer"].append(base64.b64decode(body["Record"]["Data"]))
        return JSONResponse({"RecordId": str(uuid.uuid4())})
    if operation == "PutRecordBatch" and name in streams:
        responses = []
        for record in body["Records"]:
            streams[name]["buffer"].append(base64.b64decode(record["Data"]))
            responses.append({"RecordId": str(uuid.uuid4())})
        return JSONResponse({"FailedPutCount": 0, "RequestResponses": responses})
    return JSONResponse(
        {"__type": "ResourceNotFoundException", "message": f"{operation}: {name}"},
        status_code=400,
    )


app = Starlette(routes=[Route("/", dispatch, methods=["POST"])])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
