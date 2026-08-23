# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["boto3", "botocore[crt]"]
# ///
"""Invoke the asynchronous KYC endpoint and wait for the outcome.

An asynchronous endpoint returns a location, not a result, so a verification is a
submission plus a wait. There are two ways to wait, and this is both of them: poll the
status table the notification subscriber writes, which is what an application would do,
or poll the output object in S3 directly.

Usage:
  uv run aws/invoke.py --key probes/subject_000/selfie.jpg --claim subject_000
  uv run aws/invoke.py --key probes/x.jpg --via s3
  uv run aws/invoke.py --compare a.jpg b.jpg --explain
"""

import argparse
import contextlib
import json
import os
import time
import uuid
from decimal import Decimal
from urllib.parse import urlparse

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
NAME = os.environ.get("ENDPOINT_NAME", "ch07-kyc-face")
BUCKET = os.environ["IMAGES_BUCKET"]
STATUS_TABLE = os.environ.get("STATUS_TABLE", "")


def submit(payload: dict) -> tuple[str, str]:
    """Stage the request in S3 and hand its location to the endpoint.

    The inference id is chosen here rather than left to SageMaker, because it is the
    handle the caller polls with. An application would mint it when the applicant
    presses submit and hold it in the session.
    """
    verification_id = str(uuid.uuid4())
    s3 = boto3.client("s3", region_name=REGION)
    key = f"async-in/{verification_id}.json"
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(payload).encode())
    response = boto3.client(
        "sagemaker-runtime", region_name=REGION
    ).invoke_endpoint_async(
        EndpointName=NAME,
        InputLocation=f"s3://{BUCKET}/{key}",
        InferenceId=verification_id,
        ContentType="application/json",
    )
    return verification_id, response["OutputLocation"]


def _plain(value):
    """Undo DynamoDB's Decimal numbers so the answer prints as it was written."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return [_plain(v) for v in value] if isinstance(value, list) else value


def await_status(verification_id: str, timeout: int = 900) -> dict:
    """Poll the status table until the subscriber has written this verification.

    This is the path a web or mobile onboarding flow takes: one row, one key, no view
    of the results bucket and no knowledge that a queue exists.
    """
    table = boto3.resource("dynamodb", region_name=REGION).Table(STATUS_TABLE)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if item := table.get_item(Key={"verification_id": verification_id}).get("Item"):
            if item["status"] == "failed":
                raise SystemExit(f"inference failed: {item.get('failure_reason', '')}")
            return _plain(item.get("result", item))
        time.sleep(5)
    raise TimeoutError(f"no status row for {verification_id} after {timeout}s")


def collect(location: str, timeout: int = 900) -> dict:
    """Poll the output location until the answer lands, or give up.

    The first call after the endpoint has scaled to zero waits for an instance to
    start, so the timeout is generous by design rather than by accident.
    """
    s3 = boto3.client("s3", region_name=REGION)
    parsed = urlparse(location)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    deadline = time.time() + timeout
    # SageMaker writes a failure beside the answer under the configured failure path,
    # with -error appended to the stem: async-out/<id>.out -> async-fail/<id>-error.out.
    failure_key = key.replace("async-out/", "async-fail/").replace(".out", "-error.out")
    while time.time() < deadline:
        with contextlib.suppress(s3.exceptions.NoSuchKey):
            return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        try:  # a failed inference lands here instead, and is not a timeout
            body = s3.get_object(Bucket=bucket, Key=failure_key)["Body"].read()
            raise SystemExit(f"inference failed:\n{body.decode()[:600]}")
        except s3.exceptions.NoSuchKey:
            time.sleep(5)
    raise TimeoutError(f"no result at {location} after {timeout}s")


def main() -> None:
    """Submit one verification or comparison and print the answer."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", help="probe object key, for a 1:N match")
    ap.add_argument("--claim", help="the subject the probe claims to be")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--explain", action="store_true")
    ap.add_argument(
        "--enrol",
        nargs="+",
        metavar="KEY",
        help="bulk-enrol these object keys in one batched GPU pass",
    )
    ap.add_argument(
        "--via",
        choices=("status", "s3"),
        default="status" if STATUS_TABLE else "s3",
        help="wait on the DynamoDB status row (default) or on the S3 output object",
    )
    a = ap.parse_args()

    if a.enrol:
        payload = {"op": "enrol", "keys": a.enrol}
    elif a.compare:
        payload = {
            "op": "compare",
            "a": a.compare[0],
            "b": a.compare[1],
            "explain": a.explain,
        }
    elif a.key:
        payload = {"op": "match", "key": a.key, "claim": a.claim}
    else:
        ap.error("pass --key, --compare, or --enrol")

    if a.via == "status" and not STATUS_TABLE:
        ap.error("--via status needs STATUS_TABLE set")

    verification_id, location = submit(payload)
    print("queued ->", location)
    print("verification id:", verification_id)
    answer = await_status(verification_id) if a.via == "status" else collect(location)
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
