# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlflow>=3.10,<4", "sagemaker-mlflow", "boto3"]
# ///
"""Package a registered challenger model into model.tar.gz on S3 for the serving stack.

Pulls the registered artifact from the MLflow App, repacks just the files the BYOC
container serves at the tar root, uploads it under ch02/, and prints the s3:// URL
on stdout (progress goes to stderr). template.yaml consumes that URL as ModelDataUrl.

Env: MLFLOW_TRACKING_ARN, ARTIFACT_BUCKET (required)
     MLFLOW_MODEL_PATH (default: the latest registered credit-challenger version)
     REGISTERED_MODEL (default credit-challenger)
     MODEL_FILES (default "challenger.cbm feature_spec.json")
     ENDPOINT_NAME (default ch02-challenger-byoc; only names the S3 key)
"""

import glob
import os
import sys
import tarfile

import boto3
import mlflow
from mlflow import MlflowClient

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ARN = os.environ["MLFLOW_TRACKING_ARN"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
REGISTERED_MODEL = os.environ.get("REGISTERED_MODEL", "credit-challenger")
MODEL_PATH = os.environ.get("MLFLOW_MODEL_PATH")  # empty -> resolve latest below
MODEL_FILES = os.environ.get("MODEL_FILES", "challenger.cbm feature_spec.json").split()
NAME = os.environ.get("ENDPOINT_NAME", "ch02-challenger-byoc")

mlflow.set_tracking_uri(ARN)
client = MlflowClient()

# Resolve the registered version to package, then download its SOURCE RUN's artifacts: the
# run carries the raw serving files (MODEL_FILES) that train.py logs alongside the native
# MLflow flavor, since that flavor does not itself bundle challenger.cbm / feature_spec.json.
# Default to the highest registered version (override MLFLOW_MODEL_PATH=models:/NAME/VER).
if MODEL_PATH:
    name, version = MODEL_PATH.replace("models:/", "").split("/")[:2]
    mv = client.get_model_version(name, version)
else:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if not versions:
        raise SystemExit(f"no registered versions of {REGISTERED_MODEL}")
    mv = max(versions, key=lambda v: int(v.version))
    print(
        "resolved latest registered version:",
        f"models:/{REGISTERED_MODEL}/{mv.version}",
        file=sys.stderr,
    )

local_dir = mlflow.artifacts.download_artifacts(run_id=mv.run_id)
print("downloaded run", mv.run_id, "->", local_dir, file=sys.stderr)

# Repack just the files the BYOC container serves, at the tar root (SageMaker unpacks model.tar.gz into /opt/ml/model).
tar_path = "/tmp/byoc-model.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    for fname in MODEL_FILES:
        matches = glob.glob(os.path.join(local_dir, "**", fname), recursive=True)
        if not matches:
            raise SystemExit(
                f"{fname} not found in {MODEL_PATH}; this version has no raw serving "
                f"files (e.g. an opaque-pyfunc log). Pin a version that does via "
                f"MLFLOW_MODEL_PATH."
            )
        tar.add(matches[0], arcname=fname)
        print("packed", matches[0], file=sys.stderr)

s3 = boto3.client("s3", region_name=REGION)
key = f"ch02/registry-deploy/{NAME}/model.tar.gz"
s3.upload_file(tar_path, BUCKET, key)
model_data = f"s3://{BUCKET}/{key}"
print("uploaded", model_data, file=sys.stderr)

# The URL, and only the URL, on stdout so the Makefile can capture it as ModelDataUrl.
print(model_data)
