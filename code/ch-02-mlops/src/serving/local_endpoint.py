# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "boto3"]
# ///
"""Serve the BYOC container as a SageMaker LOCAL endpoint (Mode.LOCAL_CONTAINER).

The same custom container run through SDK v3 ModelBuilder's local mode, with the trained artifact at /opt/ml/model; swapping to Mode.SAGEMAKER_ENDPOINT deploys the identical image serverless. The image is built locally (not in ECR) and ModelBuilder pulls the serving image, so this runs with SM_OFFLINE=1: it lazily stubs the SDK's account, role, and image-pull calls (see sagemaker_offline), needing no account. The credentialed counterpart is the cloud endpoint (Mode.SAGEMAKER_ENDPOINT). On an ARM Mac the amd64 image runs under emulation, so first /ping is slow.

Usage:
  SM_OFFLINE=1 MODEL_IMAGE=<local-tag> SAGEMAKER_ROLE_ARN=<any-arn> MODEL_PATH=runs-local/model \
    uv run src/serving/local_endpoint.py
"""

import json
import os
import shutil
import tempfile

from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer

# Offline only: with no account there is no STS or IAM to reach, so lazily stub the SDK's
# account, role, and image-pull calls. The native path leaves the SDK untouched.
if os.environ.get("SM_OFFLINE") == "1":
    from sagemaker_offline import use_local_stubs

    use_local_stubs()

IMAGE = os.environ["MODEL_IMAGE"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
MODEL_PATH = os.environ.get("MODEL_PATH", "runs-local/model")

# Local mode mounts <model_path>/code at /opt/ml/model, so stage the trained artifacts under a code/ dir at an absolute path (a relative path is read as a Docker volume name).
staging = tempfile.mkdtemp()
code = os.path.join(staging, "code")
os.makedirs(code)
for fname in os.listdir(MODEL_PATH):
    src = os.path.join(MODEL_PATH, fname)
    if os.path.isfile(src):
        shutil.copy(src, code)

# one application, the container's request contract (a record it scores)
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
schema = SchemaBuilder(sample_input=sample, sample_output={"pd": [0.5]})

builder = ModelBuilder(
    image_uri=IMAGE,
    model_server=ModelServer.MMS,  # the generic serve/ping/invocations runner
    model_path=staging,
    schema_builder=schema,
    role_arn=ROLE,
    mode=Mode.LOCAL_CONTAINER,
)
builder.build()
endpoint = builder.deploy_local(wait=True, container_timeout_in_seconds=1200)
resp = endpoint.invoke(body=json.dumps(sample), content_type="application/json")
print("prediction:", resp.body.read().decode() if hasattr(resp, "body") else resp)
endpoint.delete()
