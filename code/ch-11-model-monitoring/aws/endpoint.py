# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["boto3", "botocore[crt]", "pandas"]
# ///
"""Deploy the scorecard as a real-time endpoint with Data Capture, for live monitoring.

The chapter's everyday monitoring path is the Batch Transform pipeline (pipeline.py).
This is the real-time counterpart: a SageMaker endpoint serving the same train/serve
image, with DataCaptureConfig turned on so every request and response is logged to S3
as JSON Lines. Point run_monitor.py --capture at that S3 prefix and the same PSI + SHAP
monitors run over live traffic instead of a batch, no rescoring.

It packages the locally trained scorecard into model.tar.gz, creates the model, an
endpoint config with capture enabled (100% sampling, inputs and outputs, CSV), and the
endpoint; then it sends the current batch through to seed some capture and prints the S3
prefix to monitor. Needs the image in ECR (make image) first.

Env: IMAGE_URI, SAGEMAKER_ROLE_ARN, ARTIFACT_BUCKET, AWS_DEFAULT_REGION (default
     us-east-1), MODEL_DIR (default ../runs-local/model), DATA_DIR (default
     ../data/generated), INSTANCE_TYPE (default ml.m5.large), ENDPOINT (default
     ch11-scorecard).

Usage:
  IMAGE_URI=... SAGEMAKER_ROLE_ARN=... ARTIFACT_BUCKET=... PYTHONPATH=../src uv run endpoint.py
"""

import os
import tarfile
import time
from pathlib import Path

import boto3
import pandas as pd
from model import FEATURES

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
IMAGE = os.environ["IMAGE_URI"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "../runs-local/model"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "../data/generated"))
INSTANCE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
ENDPOINT = os.environ.get("ENDPOINT", "ch11-scorecard")
PREFIX = "ch11/endpoint"


def package_model() -> str:
    """Tar the trained scorecard and upload it; return the S3 model-data URL."""
    tar = Path("/tmp/ch11-model.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(MODEL_DIR / "scorecard.cbm", arcname="scorecard.cbm")
    key = f"{PREFIX}/model/model.tar.gz"
    boto3.client("s3", region_name=REGION).upload_file(str(tar), BUCKET, key)
    return f"s3://{BUCKET}/{key}"


def main() -> None:
    """Create the capture-enabled endpoint, seed it with the current batch, print the prefix."""
    sm = boto3.client("sagemaker", region_name=REGION)
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    model_data = package_model()
    capture_s3 = f"s3://{BUCKET}/{PREFIX}/capture"

    name = f"ch11-scorecard-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    sm.create_model(
        ModelName=name,
        PrimaryContainer={"Image": IMAGE, "ModelDataUrl": model_data},
        ExecutionRoleArn=ROLE,
    )
    # DataCaptureConfig logs every request and response to S3 for the monitors to read
    sm.create_endpoint_config(
        EndpointConfigName=name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": name,
                "InstanceType": INSTANCE,
                "InitialInstanceCount": 1,
            }
        ],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": capture_s3,
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"CsvContentTypes": ["text/csv"]},
        },
    )
    exists = ENDPOINT in [
        e["EndpointName"] for e in sm.list_endpoints(NameContains=ENDPOINT)["Endpoints"]
    ]
    if exists:
        sm.update_endpoint(EndpointName=ENDPOINT, EndpointConfigName=name)
    else:
        sm.create_endpoint(EndpointName=ENDPOINT, EndpointConfigName=name)
    print(f"waiting for {ENDPOINT} to be InService...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=ENDPOINT)

    # seed capture: send the current batch through in small invocations
    current = pd.read_csv(DATA_DIR / "current.csv")
    for start in range(0, len(current), 100):
        body = (
            current[FEATURES]
            .iloc[start : start + 100]
            .to_csv(header=False, index=False)
        )
        rt.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv", Body=body)
    print(f"endpoint {ENDPOINT} InService; sent {len(current)} rows")
    print(f"capture landing under: {capture_s3}/{ENDPOINT}/AllTraffic/")
    print(
        f"monitor it:  uv run src/run_monitor.py --capture {capture_s3}/{ENDPOINT}/AllTraffic/"
    )


if __name__ == "__main__":
    main()
