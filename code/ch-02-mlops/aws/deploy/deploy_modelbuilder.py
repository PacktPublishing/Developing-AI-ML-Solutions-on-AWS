# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sagemaker>=3,<4", "mlflow>=3.10,<4", "sagemaker-mlflow", "boto3", "pandas"]
# ///
"""Auto-deploy a registered model from the MLflow App with SDK v3 ModelBuilder.

This is the bridge from "Logged" to "Deployable": an MLflow-registered pyfunc is
not a SageMaker-deployable model on its own. ModelBuilder pulls it from the App
registry, chooses a serving image, installs the model's own requirements, and
builds a deployable SageMaker model — then deploys it to an endpoint. This is the
"pin the environment in the model" path (the environment is rebuilt from the
model's requirements), the counterpart to the BYOC "same image trains and serves"
path used elsewhere in the chapter.

Deploys serverless so no instance quota is needed. Env:
  MLFLOW_TRACKING_ARN, SAGEMAKER_ROLE_ARN (required)
  MLFLOW_MODEL_PATH (default models:/credit-challenger/2)
"""

import os

import pandas as pd
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.serverless import ServerlessInferenceConfig

ARN = os.environ["MLFLOW_TRACKING_ARN"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
MODEL_PATH = os.environ.get("MLFLOW_MODEL_PATH", "models:/credit-challenger/2")

sample_input = pd.DataFrame(
    [
        {
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
    ]
)
sample_output = [0.5]

HERE = os.path.dirname(os.path.abspath(__file__))

builder = ModelBuilder(
    mode=Mode.SAGEMAKER_ENDPOINT,
    schema_builder=SchemaBuilder(
        sample_input=sample_input, sample_output=sample_output
    ),
    role_arn=ROLE,
    model_metadata={"MLFLOW_MODEL_PATH": MODEL_PATH, "MLFLOW_TRACKING_ARN": ARN},
    # A custom pyfunc (fastwoe / xgboost) isn't a native MLflow flavor, so skip
    # ModelBuilder's auto dependency detection and hand it the requirements —
    # otherwise it tries to introspect the pickle and cannot import our classes.
    dependencies={
        "auto": False,
        "requirements": os.path.join(HERE, "mb-requirements.txt"),
    },
)

builder.build()
print("built deployable model from", MODEL_PATH)

# v3: deploy from the builder, serverless via inference_config (no instance quota).
predictor = builder.deploy(
    endpoint_name=os.environ.get("ENDPOINT_NAME", "ch02-challenger-mb"),
    inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=3072, max_concurrency=2
    ),
)
print("deployed serverless endpoint:", getattr(predictor, "endpoint_name", predictor))

# The SchemaBuilder gave the predictor a serializer that matches the container,
# so scoring is a plain call — the same two applicants used everywhere else.
print("prediction:", predictor.predict(sample_input))
