# /// script
# dependencies = ["boto3", "ollama"]
# ///
"""The batch-inference seam: real Bedrock on AWS, a from-source shim locally.

get_bedrock() returns a control-plane client exposing create_model_invocation_job
and get_model_invocation_job. On AWS that is boto3's bedrock client. Locally it is
LocalBedrockBatch, which -- like Amazon Bedrock and like LocalStack's own emulator
-- runs the job as a background thread: read the JSONL shards from S3, invoke the
model on each record (through the Ollama-backed model seam), and write the
responses back as .jsonl.out. The submit/poll/collect client code never changes.
"""

import json
import os
import threading
import uuid

import boto3
from botocore.exceptions import ClientError

from models import get_runtime


def s3_client():
    """Return an S3 client -- the local S3Proxy when running locally, else real S3."""
    endpoint = os.environ.get("S3_ENDPOINT")
    if endpoint:
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("S3_IDENTITY", "local"),
            aws_secret_access_key=os.environ.get("S3_CREDENTIAL", "localsecret"),
            region_name=os.environ.get("BEDROCK_REGION", "us-east-1"),
            config=boto3.session.Config(s3={"addressing_style": "path"}),
        )
    return boto3.client("s3", region_name=os.environ.get("BEDROCK_REGION", "us-east-1"))


def _split_uri(uri: str) -> tuple[str, str]:
    """Split s3://bucket/prefix into (bucket, prefix)."""
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    return bucket, prefix


class LocalBedrockBatch:
    """A from-source stand-in for Bedrock batch inference, backed by the model seam."""

    def __init__(self) -> None:
        """Open the model runtime and an S3 client, and start an empty job table."""
        self._runtime = get_runtime()
        self._s3 = s3_client()
        self._jobs: dict[str, dict] = {}

    def create_model_invocation_job(
        self, jobName, modelId, inputDataConfig, outputDataConfig, roleArn=None, **_
    ) -> dict:
        """Register the job and run it on a background thread, as Bedrock does."""
        job_id = str(uuid.uuid4())
        arn = f"arn:aws:bedrock:local:000000000000:model-invocation-job/{job_id}"
        self._jobs[arn] = {
            "jobArn": arn,
            "jobName": jobName,
            "modelId": modelId,
            "status": "Submitted",
            "inputDataConfig": inputDataConfig,
            "outputDataConfig": outputDataConfig,
        }
        threading.Thread(target=self._run, args=(arn,), daemon=True).start()
        return {"jobArn": arn}

    def get_model_invocation_job(self, jobIdentifier, **_) -> dict:
        """Return the current job details by ARN or bare job id, including status."""
        job = self._jobs.get(jobIdentifier) or next(
            (j for arn, j in self._jobs.items() if arn.split("/")[-1] == jobIdentifier),
            None,
        )
        if job is None:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ResourceNotFoundException",
                        "Message": "The specified resource Amazon Resource Name"
                        " (ARN) was not found. Check the Amazon Resource Name"
                        " (ARN) and try your request again.",
                    }
                },
                "GetModelInvocationJob",
            )
        return job

    def _run(self, arn: str) -> None:
        """Read the input shards, invoke the model per record, write .jsonl.out."""
        job = self._jobs[arn]
        job["status"] = "InProgress"
        in_bucket, in_prefix = _split_uri(
            job["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
        )
        out_bucket, out_prefix = _split_uri(
            job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        )
        model = job["modelId"].split("/")[-1]
        try:
            listing = self._s3.list_objects_v2(Bucket=in_bucket, Prefix=in_prefix)
            for obj in listing.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".jsonl"):
                    continue
                body = (
                    self._s3.get_object(Bucket=in_bucket, Key=key)["Body"]
                    .read()
                    .decode("utf-8")
                )
                out_lines = [
                    json.dumps(self._invoke(model, json.loads(line)))
                    for line in body.splitlines()
                    if line.strip()
                ]
                out_key = f"{out_prefix.rstrip('/')}/{arn.split('/')[-1]}/{key.split('/')[-1]}.out"
                self._s3.put_object(
                    Bucket=out_bucket,
                    Key=out_key,
                    Body="\n".join(out_lines).encode("utf-8"),
                )
            job["status"] = "Completed"
        except Exception as exc:  # noqa: BLE001 - surface the failure in the job status like Bedrock
            job["status"] = "Failed"
            job["message"] = str(exc)

    def _invoke(self, model: str, record: dict) -> dict:
        """Invoke the model on one record's Converse modelInput; wrap as Bedrock does."""
        mi = record["modelInput"]
        resp = self._runtime.converse(
            modelId=model,
            messages=mi["messages"],
            system=mi.get("system"),
            inferenceConfig=mi.get("inferenceConfig"),
        )
        return {"recordId": record["recordId"], "modelInput": mi, "modelOutput": resp}


def get_bedrock():
    """Return a Bedrock control-plane client: real boto3, or the local batch shim."""
    if os.environ.get("BEDROCK_LOCAL") == "1":
        return LocalBedrockBatch()
    return boto3.client(
        "bedrock", region_name=os.environ.get("BEDROCK_REGION", "us-east-1")
    )
