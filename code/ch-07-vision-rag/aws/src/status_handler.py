"""Turn an asynchronous inference notification into a row the application can poll.

SNS has no DynamoDB protocol, so a subscriber sits between them. It receives the
endpoint's completion notice, reads the answer out of the results bucket, and writes
one item keyed by the inference id. The applicant's session polls that item and never
sees S3, the queue, or the endpoint.
"""

import json
import logging
import os
import time
from decimal import Decimal
from urllib.parse import urlparse

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

TABLE = os.environ["STATUS_TABLE"]
TTL_SECONDS = int(os.environ.get("STATUS_TTL_SECONDS", "86400"))

# A verification answer is small, so the whole body is stored rather than a summary of
# it. Anything larger than a few kilobytes belongs behind a presigned URL instead.
MAX_BODY_BYTES = 32_000


def _read(location: str) -> object:
    """Return the object at an s3:// URI, parsed if it is JSON."""
    parsed = urlparse(location)
    body = (
        boto3.client("s3")
        .get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
        .read(MAX_BODY_BYTES)
    )
    try:
        # DynamoDB has no float type, so decode the numbers straight into Decimal
        # rather than converting them afterwards.
        return json.loads(body, parse_float=Decimal)
    except json.JSONDecodeError:
        return body.decode(errors="replace")


def _item(notice: dict) -> dict:
    """Build the status row for one SageMaker notification.

    The fields are invocationStatus, requestParameters, responseParameters, inferenceId
    and eventTime. A success and a failure name their body differently: responseParameters
    carries outputLocation on the way through, and failureLocation instead when the model
    returned an error, so the two are read from different keys rather than derived from
    one another.
    """
    completed = notice.get("invocationStatus") == "Completed"
    response = notice.get("responseParameters", {})
    location = response.get("outputLocation" if completed else "failureLocation", "")
    item = {
        "verification_id": notice["inferenceId"],
        "status": "completed" if completed else "failed",
        "endpoint": notice.get("requestParameters", {}).get("endpointName", ""),
        "input_location": notice.get("requestParameters", {}).get("inputLocation", ""),
        "output_location": location,
        "updated_at": notice.get("eventTime", ""),
        "expires_at": int(time.time()) + TTL_SECONDS,
    }
    if not completed:
        item["failure_reason"] = notice.get("failureReason", "")
    if location:
        try:
            item["result"] = _read(location)
        except Exception as exc:  # the row is still worth writing without the body
            log.warning("could not read %s: %s", location, exc)
            item["read_error"] = str(exc)
    return item


def lambda_handler(event, context):
    """Write one status row per notification the topic delivered."""
    table = boto3.resource("dynamodb").Table(TABLE)
    written = 0
    for record in event.get("Records", []):
        notice = json.loads(record["Sns"]["Message"])
        item = _item(notice)
        if item["status"] == "failed":
            log.warning("failed notification: %s", record["Sns"]["Message"])
        table.put_item(Item=item)
        log.info("%s -> %s", item["verification_id"], item["status"])
        written += 1
    return {"written": written}
