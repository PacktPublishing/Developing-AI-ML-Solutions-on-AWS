# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "pandas"]
# ///
"""Replay the transaction file into the stream, one PutRecord per event.

This stands in for the card processor: every transaction becomes one record on
the transactions stream the moment it happens. The partition key is the user
id, and that choice is doing real work — Kinesis hashes the key to pick a
shard, so all of one user's events land on the same shard in order. The
consumer can then reason about "the previous transaction of this user"
without cross-shard coordination.

The loop also measures what a producer cares about: per-call latency and
throttles. Run it against the local container and against a real stream and
compare — the API calls are identical, only the endpoint differs.

Usage:
  uv run streaming/producer.py --limit 2000
"""

import argparse
import json
import statistics
import time

import pandas as pd
from botocore.exceptions import ClientError
from common import TRANSACTIONS_STREAM, ensure_stream, kinesis_client


def main() -> None:
    """Send transactions to the stream and report producer-side metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--file", default="data/stream/transactions.csv")
    args = parser.parse_args()

    kinesis = kinesis_client()
    ensure_stream(kinesis, TRANSACTIONS_STREAM)

    df = pd.read_csv(args.file, nrows=args.limit)
    latencies, sent, throttled = [], 0, 0
    t0 = time.perf_counter()
    for row in df.itertuples(index=False):
        payload = json.dumps(row._asdict())
        t = time.perf_counter()
        try:
            kinesis.put_record(
                StreamName=TRANSACTIONS_STREAM,
                Data=payload,
                PartitionKey=row.user_id,
            )
            sent += 1
        except ClientError as e:
            if e.response["Error"]["Code"] == "ProvisionedThroughputExceededException":
                throttled += 1
            else:
                raise
        latencies.append((time.perf_counter() - t) * 1000)
    elapsed = time.perf_counter() - t0

    p95 = statistics.quantiles(latencies, n=20)[18]
    print(f"{sent} records in {elapsed:.1f}s = {sent / elapsed:.0f} rec/s")
    print(
        f"latency p50={statistics.median(latencies):.1f}ms p95={p95:.1f}ms "
        f"throttled={throttled}"
    )


if __name__ == "__main__":
    main()
