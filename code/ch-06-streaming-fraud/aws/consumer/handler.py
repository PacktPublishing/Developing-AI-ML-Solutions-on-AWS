"""Score the transaction stream on AWS: the cloud twin of src/streaming/consumer.py.

The Kinesis event source mapping replaces the local polling loop, so this
handler is only the loop body: invoke the endpoint, write the block, forward
the decision. The endpoint returns a 0-1000 score, and the two cuts still ship
from training (BLOCK_CUT and REVIEW_CUT, copied from model_meta.json) and
are never re-derived here. Partial batch failures are reported per record, so
one poison message does not stall the shard.
"""

import base64
import json
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
BLOCK_CUT = float(os.environ["BLOCK_CUT"])
REVIEW_CUT = float(os.environ["REVIEW_CUT"])
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
            score = json.loads(response["Body"].read())["scores"][0]
            # The banding of src/scoring/calibration.py, the same ordered check
            # the local consumer runs. First match wins.
            if score >= BLOCK_CUT:
                decision = "block"
            elif score >= REVIEW_CUT:
                decision = "investigate"
            else:
                decision = "approve"

            # The fast path: only a block gets written; a miss lets the payment
            # through, which is what an investigate does too, the case being
            # reviewed after the payment completes.
            if decision == "block":
                table.put_item(
                    Item={
                        "transaction_id": txn["transaction_id"],
                        "user_id": txn["user_id"],
                        "score": int(score),
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
                        "score": int(score),
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
