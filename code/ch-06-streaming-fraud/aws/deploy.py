# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "sagemaker-serve", "botocore[crt]", "boto3"]
# ///
"""Deploy the fraud model behind the SageMaker serving contract.

Two modes, one image, one model. The default rehearses the deploy in
SageMaker local mode (ModelBuilder, Mode.LOCAL_CONTAINER): real role, real
ECR image, but the container runs on this machine and serves on :8080.
SAGEMAKER_SERVERLESS=1 runs the real deploy instead — the chapter-02 BYOC
choreography: model.tar.gz to S3, then model + endpoint config + endpoint
with a ServerlessConfig. The streaming consumer's invoke_endpoint call works
against either, unchanged.

Env: IMAGE_URI, SAGEMAKER_ROLE_ARN, ARTIFACT_BUCKET (required; the Makefile
     fills them from the stack outputs); ENDPOINT_NAME (default fraud-scoring).

Usage:
  uv run aws/deploy.py                        # local mode, on :8080
  SAGEMAKER_SERVERLESS=1 uv run aws/deploy.py # the serverless deploy
"""

import os
import tarfile

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
IMAGE_URI = os.environ["IMAGE_URI"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
NAME = os.environ.get("ENDPOINT_NAME", "fraud-scoring")
SERVERLESS = os.environ.get("SAGEMAKER_SERVERLESS") == "1"

CHAPTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deploy_local() -> None:
    """Run the serving image as a local-mode endpoint on :8080."""
    from sagemaker.serve.mode.function_pointers import Mode
    from sagemaker.serve.model_builder import ModelBuilder
    from sagemaker.serve.utils.types import ModelServer

    builder = ModelBuilder(
        mode=Mode.LOCAL_CONTAINER,
        image_uri=IMAGE_URI,
        model_path=f"{CHAPTER_DIR}/artifacts",
        role_arn=ROLE,
        # explicit staging path — without it the SDK creates an account-default bucket
        s3_model_data_url=f"s3://{BUCKET}/ch06/{NAME}/local-mode",
        # TORCHSERVE selects the generic `docker run <image> serve` runner —
        # the plain container contract this chapter's image implements
        model_server=ModelServer.TORCHSERVE,
    )
    builder.build()
    builder.deploy(endpoint_name=NAME)
    print(f"local-mode endpoint {NAME} serving on http://localhost:8080")


def deploy_serverless() -> None:
    """Create the Serverless Inference endpoint from the same image and model."""
    # 1. Package the artifacts at the tar root; SageMaker unpacks model.tar.gz
    #    into /opt/ml/model, exactly where the local runs mount artifacts/.
    tar_path = "/tmp/ch06-model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for fname in ("model.cbm", "model_meta.json"):
            tar.add(os.path.join(CHAPTER_DIR, "artifacts", fname), arcname=fname)
            print("packed", fname)

    s3 = boto3.client("s3", region_name=REGION)
    key = f"ch06/{NAME}/model.tar.gz"
    s3.upload_file(tar_path, BUCKET, key)
    model_data = f"s3://{BUCKET}/{key}"
    print("uploaded", model_data)

    # 2. Model + serverless endpoint on the chapter's own image. Recreate from
    #    scratch so a re-run picks up a new model artifact.
    sm = boto3.client("sagemaker", region_name=REGION)
    for delete in (
        lambda: sm.delete_endpoint(EndpointName=NAME),
        lambda: sm.delete_endpoint_config(EndpointConfigName=NAME),
        lambda: sm.delete_model(ModelName=NAME),
    ):
        try:
            delete()
        except sm.exceptions.ClientError:
            pass

    sm.create_model(
        ModelName=NAME,
        ExecutionRoleArn=ROLE,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": model_data,
            "Mode": "SingleModel",
        },
    )
    sm.create_endpoint_config(
        EndpointConfigName=NAME,
        ProductionVariants=[
            {
                "VariantName": "main",
                "ModelName": NAME,
                "ServerlessConfig": {"MemorySizeInMB": 2048, "MaxConcurrency": 5},
            }
        ],
    )
    sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
    print("creating endpoint", NAME, "...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME)
    print("InService:", NAME)


if __name__ == "__main__":
    deploy_serverless() if SERVERLESS else deploy_local()
