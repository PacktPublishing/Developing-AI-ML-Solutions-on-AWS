# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mlflow>=3.10,<4", "sagemaker-mlflow", "boto3"]
# ///
"""Deploy a registered model from the MLflow App on the BYOC image.

The "pin the environment in the image" counterpart to deploy_modelbuilder.py.
Instead of letting ModelBuilder repackage the model onto a heavy serving DLC
(which overran the 3072 MB serverless quota), this pulls the model artifact from
the App registry and serves it on the same small custom container that trained
it — the image already proven to give byte-exact parity. It fits the serverless
memory quota and needs no dependency resolution at deploy time.

Env: MLFLOW_TRACKING_ARN, SAGEMAKER_ROLE_ARN, ARTIFACT_BUCKET, IMAGE_URI (required)
     MLFLOW_MODEL_PATH (default models:/credit-challenger/3)
     MODEL_FILES (default "challenger.ubj feature_spec.json")
"""

import glob
import os
import tarfile

import boto3
import mlflow

# -------------------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------------------
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ARN = os.environ["MLFLOW_TRACKING_ARN"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
IMAGE_URI = os.environ["IMAGE_URI"]
MODEL_PATH = os.environ.get("MLFLOW_MODEL_PATH", "models:/credit-challenger/3")
MODEL_FILES = os.environ.get("MODEL_FILES", "challenger.ubj feature_spec.json").split()
NAME = os.environ.get("ENDPOINT_NAME", "ch02-challenger-byoc")

# -------------------------------------------------------------------------------
# Artifact download
# -------------------------------------------------------------------------------
# 1. Pull the registered model's artifacts from the App registry.
mlflow.set_tracking_uri(ARN)
local_dir = mlflow.artifacts.download_artifacts(MODEL_PATH)
print("downloaded", MODEL_PATH, "->", local_dir)

# -------------------------------------------------------------------------------
# Repackage and upload
# -------------------------------------------------------------------------------
# 2. Repackage just the files the BYOC container serves, at the tar root
#    (SageMaker unpacks model.tar.gz into /opt/ml/model).
tar_path = "/tmp/byoc-model.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    for fname in MODEL_FILES:
        matches = glob.glob(os.path.join(local_dir, "**", fname), recursive=True)
        if not matches:
            raise SystemExit(f"{fname} not found under {local_dir}")
        tar.add(matches[0], arcname=fname)
        print("packed", matches[0])

s3 = boto3.client("s3", region_name=REGION)
key = f"ch02/registry-deploy/{NAME}/model.tar.gz"
s3.upload_file(tar_path, BUCKET, key)
model_data = f"s3://{BUCKET}/{key}"
print("uploaded", model_data)

# -------------------------------------------------------------------------------
# Serverless endpoint
# -------------------------------------------------------------------------------
# 3. Model + serverless endpoint on the BYOC image (fits the 3072 MB quota).
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
            "ServerlessConfig": {"MemorySizeInMB": 3072, "MaxConcurrency": 2},
        }
    ],
)
sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
print("creating endpoint", NAME, "...")
sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME)
print("InService:", NAME)
