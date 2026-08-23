"""Registration trigger: hand new identity photos to the asynchronous embedding endpoint.

An upload under `registered/{subject}/` fires this function, which does no inference of
its own. It stages a request in S3 and calls the SageMaker endpoint asynchronously, so
every embedding in the system, registration and verification alike, is produced by the same
model on the same GPU. The function is a few lines of boto3, which is why it ships as a
zip rather than a container.
"""

import json
import logging
import os
import uuid

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

ENDPOINT = os.environ["SAGEMAKER_ENDPOINT"]
BUCKET = os.environ["IMAGES_S3_BUCKET"]


def _keys(event) -> list[str]:
    """Return the object keys from an S3 trigger or a direct {"keys": [...]} call."""
    if "Records" in event:
        return [r["s3"]["object"]["key"] for r in event["Records"]]
    return event.get("keys", [])


def lambda_handler(event, context):
    """Queue the uploaded photos for embedding on the endpoint."""
    keys = _keys(event)
    if not keys:
        return {"queued": 0}

    s3 = boto3.client("s3")
    # The same id names the staged request and the notification, so an registration can
    # be followed through the status table exactly like a verification.
    request_id = str(uuid.uuid4())
    request_key = f"async-in/{request_id}.json"
    s3.put_object(
        Bucket=BUCKET,
        Key=request_key,
        Body=json.dumps({"op": "register", "keys": keys}).encode(),
    )

    # Asynchronous: the endpoint may be at zero instances, in which case this request
    # waits in its queue and wakes it. Nothing here blocks on the model starting.
    response = boto3.client("sagemaker-runtime").invoke_endpoint_async(
        EndpointName=ENDPOINT,
        InputLocation=f"s3://{BUCKET}/{request_key}",
        InferenceId=request_id,
        ContentType="application/json",
    )
    log.info("queued %d key(s) -> %s", len(keys), response["OutputLocation"])
    return {"queued": len(keys), "output": response["OutputLocation"]}
