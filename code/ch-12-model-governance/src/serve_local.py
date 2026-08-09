# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "boto3"]
# ///
"""Serve one rollout model as a SageMaker local endpoint (Mode.LOCAL_CONTAINER).

Option 2 for the models locally: instead of the compose rollout stack, champion or
challenger runs through ModelBuilder local mode, the same ModelBuilder path and image the
cloud endpoint uses, so swapping Mode.LOCAL_CONTAINER for Mode.SAGEMAKER_ENDPOINT is the
only change to reach the serverless endpoint the stack deploys. One model, deployed,
invoked once, then deleted (the serving check for a model; the champion/challenger A/B
runs on the compose stack). Needs AWS credentials, a SageMaker role, and the model's
serving image in ECR; the trained artifact is mounted from chapter 2's local run.

Usage:
  MODEL_IMAGE=<ecr-uri> SAGEMAKER_ROLE_ARN=<role> \
    MODEL_PATH=../ch-02-mlops/runs-local/model uv run src/serve_local.py
"""

import json
import os
import shutil
import tempfile

from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer

IMAGE = os.environ["MODEL_IMAGE"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
MODEL_PATH = os.environ.get("MODEL_PATH", "../ch-02-mlops/runs-local/model")

# one application, the serving container's request contract (chapter 2's schema)
sample = {
    "age": 35,
    "annual_income": 42000,
    "debt_to_income": 38.0,
    "bureau_score": 590,
    "credit_utilization": 85.0,
    "employment_length_years": 1.5,
    "loan_amount": 22000,
    "home_ownership": "RENT",
    "loan_purpose": "debt_consolidation",
    "employment_status": "self_employed",
}

# Local mode mounts <model_path>/code at /opt/ml/model, so stage the trained artifact under
# a code/ dir at an absolute path (a relative path is read as a Docker volume name).
staging = tempfile.mkdtemp()
code = os.path.join(staging, "code")
os.makedirs(code)
for fname in os.listdir(MODEL_PATH):
    src = os.path.join(MODEL_PATH, fname)
    if os.path.isfile(src):
        shutil.copy(src, code)

builder = ModelBuilder(
    image_uri=IMAGE,
    model_server=ModelServer.MMS,  # the generic serve/ping/invocations runner
    model_path=staging,
    schema_builder=SchemaBuilder(sample_input=sample, sample_output={"pd": [0.5]}),
    role_arn=ROLE,
    mode=Mode.LOCAL_CONTAINER,
)
builder.build()
endpoint = builder.deploy_local(wait=True, container_timeout_in_seconds=1200)
resp = endpoint.invoke(body=json.dumps(sample), content_type="application/json")
print("prediction:", resp.body.read().decode() if hasattr(resp, "body") else resp)
endpoint.delete()
