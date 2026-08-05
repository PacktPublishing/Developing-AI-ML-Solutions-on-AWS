# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Score the transaction stream and block the transactions that look like fraud.

Read transactions off the stream, score each against the model, and act on the
decision in two places. DynamoDB holds blocks only, keyed by transaction id:
the block lands first because that write is the product. The decisions stream
then gets every scored record, blocked or passed, for the analytics pipeline.
Fast path first, analytics second; the order in the loop body is the
architecture.

The model sits behind the SageMaker serving contract, never in-process: the
consumer calls invoke_endpoint through boto3's sagemaker-runtime client, the
local serving container (`make serve`) or the same image behind Serverless
Inference, and only the endpoint URL changes. The threshold comparison stays
on the consumer side; the endpoint returns a probability, the consumer owns
the decision.

Usage:
  uv run streaming/consumer.py
"""

import json
import statistics
import time
from decimal import Decimal

from common import (
    BLOCKS_TABLE,
    DECISIONS_STREAM,
    SCORING_ENDPOINT_NAME,
    TRANSACTIONS_STREAM,
    dynamodb_resource,
    ensure_stream,
    iter_records,
    kinesis_client,
    sagemaker_runtime_client,
)


def ensure_table(dynamodb):
    """Create the blocks table if it does not exist and return it."""
    try:
        table = dynamodb.create_table(
            TableName=BLOCKS_TABLE,
            KeySchema=[{"AttributeName": "transaction_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "transaction_id", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        table = dynamodb.Table(BLOCKS_TABLE)
    return table


def main() -> None:
    """Drain the transactions stream, scoring every record as it arrives."""
    # The training-serving contract: the threshold ships with the model and
    # is never re-derived here. The feature columns live server-side.
    with open("artifacts/model_meta.json") as f:
        threshold = json.load(f)["threshold"]

    kinesis = kinesis_client()
    ensure_stream(kinesis, TRANSACTIONS_STREAM)
    ensure_stream(kinesis, DECISIONS_STREAM)
    table = ensure_table(dynamodb_resource())
    runtime = sagemaker_runtime_client()

    scored, blocked, latencies = 0, 0, []
    for txn in iter_records(kinesis, TRANSACTIONS_STREAM):
        t = time.perf_counter()
        response = runtime.invoke_endpoint(
            EndpointName=SCORING_ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(txn),
        )
        score = json.loads(response["Body"].read())["scores"][0]
        decision = "block" if score >= threshold else "pass"
        latency_ms = (time.perf_counter() - t) * 1000

        # The fast path: only fraud gets written. The processor looks up the id;
        # a block stops the payment, a miss lets it through.
        if decision == "block":
            table.put_item(
                Item={
                    "transaction_id": txn["transaction_id"],
                    "user_id": txn["user_id"],
                    "score": Decimal(f"{score:.6f}"),
                    "latency_ms": Decimal(f"{latency_ms:.3f}"),
                }
            )

        # The analytics path: the same decision, onward to the delivery worker.
        kinesis.put_record(
            StreamName=DECISIONS_STREAM,
            Data=json.dumps(
                {
                    "transaction_id": txn["transaction_id"],
                    "user_id": txn["user_id"],
                    "merchant_id": txn["merchant_id"],
                    "amount_usd": txn["amount_usd"],
                    "score": round(score, 6),
                    "decision": decision,
                    "event_time": txn["event_time"],
                    "latency_ms": round(latency_ms, 3),
                }
            ),
            PartitionKey=txn["user_id"],
        )
        scored += 1
        blocked += decision == "block"
        latencies.append(latency_ms)

    if not scored:
        print("no records on the stream: run the producer first")
        return
    p95 = (
        statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20
        else max(latencies)
    )
    print(f"scored {scored}, blocked {blocked} ({blocked / scored:.2%})")
    print(f"endpoint latency p50={statistics.median(latencies):.1f}ms p95={p95:.1f}ms")


if __name__ == "__main__":
    main()
