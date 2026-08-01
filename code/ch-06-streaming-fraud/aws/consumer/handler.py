"""Score the transaction stream on AWS: the cloud twin of src/streaming/consumer.py.

The Kinesis event source mapping does what the local polling loop did — track
a per-shard iterator and hand over batches — so this handler is only the loop
body: invoke the endpoint, write the block, forward the decision. The three
moves and their order are identical to the local consumer; the threshold
still ships from training (as the SCORE_THRESHOLD parameter, copied from
model_meta.json) and is never re-derived here.

Partial batch failures are reported per record, so one poison message does
not stall the shard.
"""

import base64
import json
import os
from decimal import Decimal

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
THRESHOLD = float(os.environ["SCORE_THRESHOLD"])
BLOCKS_TABLE = os.environ["BLOCKS_TABLE"]
DECISIONS_STREAM = os.environ["DECISIONS_STREAM"]

runtime = boto3.client("sagemaker-runtime")
table = boto3.resource("dynamodb").Table(BLOCKS_TABLE)
kinesis = boto3.client("kinesis")


def score(event, context):
    """Score one Kinesis batch, block the fraud, forward every decision."""
    failures = []
    for record in event["Records"]:
        try:
            txn = json.loads(base64.b64decode(record["kinesis"]["data"]))
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(txn),
            )
            probability = json.loads(response["Body"].read())["scores"][0]
            decision = "block" if probability >= THRESHOLD else "pass"

            # The fast path: only fraud gets written; a miss lets the
            # payment through.
            if decision == "block":
                table.put_item(
                    Item={
                        "transaction_id": txn["transaction_id"],
                        "user_id": txn["user_id"],
                        "score": Decimal(f"{probability:.6f}"),
                    }
                )

            # The analytics path: every decision onward to Firehose.
            kinesis.put_record(
                StreamName=DECISIONS_STREAM,
                Data=json.dumps(
                    {
                        "transaction_id": txn["transaction_id"],
                        "user_id": txn["user_id"],
                        "merchant_id": txn["merchant_id"],
                        "amount_usd": txn["amount_usd"],
                        "score": round(probability, 6),
                        "decision": decision,
                        "event_time": txn["event_time"],
                    }
                ),
                PartitionKey=txn["user_id"],
            )
        except (ClientError, BotoCoreError, KeyError, ValueError):
            # a malformed record or a failed AWS call: report just this item
            # so Lambda retries it without stalling the shard
            failures.append({"itemIdentifier": record["kinesis"]["sequenceNumber"]})
    return {"batchItemFailures": failures}
