# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "boto3"]
# ///
"""Deploy, invoke and delete a model endpoint using SageMaker local mode.

ModelBuilder uses the serving image in ECR and a trained Chapter 2 artifact.
This serving check requires AWS credentials and a SageMaker role.

Usage:
  MODEL_IMAGE=<ecr-uri> SAGEMAKER_ROLE_ARN=<role>     MODEL_PATH=../ch-02-mlops/runs-local/model uv run src/serve_local.py
"""

import json
import os
import shutil
import tempfile

# The same stubs ch-02, ch-04 and ch-07 use for their local-mode paths: without
# them ModelBuilder resolves the account's default SageMaker bucket, which no
# chapter owns and a ch12-* policy cannot reach, and the run dies on a 403 from
# head_bucket before a container ever starts.
if os.environ.get("SM_OFFLINE") == "1":
    from sagemaker_offline import use_local_stubs

    use_local_stubs()

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
try:
    resp = endpoint.invoke(body=json.dumps(sample), content_type="application/json")
    print("prediction:", resp.body.read().decode() if hasattr(resp, "body") else resp)
finally:
    # The container outlives this process otherwise, holding the port and the
    # model, and a question that failed is exactly when it is left behind.
    endpoint.delete()
