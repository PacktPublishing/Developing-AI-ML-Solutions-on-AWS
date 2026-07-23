# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sagemaker>=3,<4", "boto3"]
# ///
"""SageMaker Automatic Model Tuning for the challenger (SDK v3).

The instance-based hyperparameter search: a HyperparameterTuner drives many
training jobs of the BYOC image over ranges, reads the objective from each job's
logs (the "validation_auc:" line the container prints), and returns the best. The
same serverless MLflow App captures every trial as a run, so the search is both
an AMT tuning job and an MLflow experiment.

This is the v3 form of the classic tuner: the training image is our own ECR
image, and MLFLOW_TRACKING_URI is a serverless MLflow App ARN rather than an
always-on tracking server.

Requires SageMaker training-job quota (0 on a fresh account — see aws/README).
For a search that runs without that quota, use tuning/amt.py (Syne Tune Bayesian
search running the same training code locally, tracked in the same MLflow App).

Env: IMAGE_URI, SAGEMAKER_ROLE_ARN, ARTIFACT_BUCKET, MLFLOW_TRACKING_ARN (required)
"""

import os

from sagemaker.core.parameter import ContinuousParameter, IntegerParameter
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (
    Compute,
    InputData,
    OutputDataConfig,
    StoppingCondition,
)
from sagemaker.train.tuner import HyperparameterTuner

# -------------------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------------------
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
IMAGE_URI = os.environ["IMAGE_URI"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
APP_ARN = os.environ["MLFLOW_TRACKING_ARN"]
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")

# -------------------------------------------------------------------------------
# The base trainer
# -------------------------------------------------------------------------------
# The training image and its static (non-tuned) settings, exactly as a single
# training job would run — the tuner varies the ranges below on top of this.
trainer = ModelTrainer(
    training_image=IMAGE_URI,
    role=ROLE,
    base_job_name="ch02-hpo",
    compute=Compute(
        instance_type=INSTANCE_TYPE, instance_count=1, volume_size_in_gb=30
    ),
    output_data_config=OutputDataConfig(
        s3_output_path=f"s3://{BUCKET}/ch02/hpo-output"
    ),
    hyperparameters={"monotonic": "true", "registered_model_name": "credit-challenger"},
    environment={"MLFLOW_TRACKING_URI": APP_ARN},
    stopping_condition=StoppingCondition(max_runtime_in_seconds=1800),
)

# -------------------------------------------------------------------------------
# The tuner
# -------------------------------------------------------------------------------
tuner = HyperparameterTuner(
    model_trainer=trainer,
    objective_metric_name="validation_auc",
    objective_type="Maximize",
    hyperparameter_ranges={
        "max_depth": IntegerParameter(3, 8),
        "n_estimators": IntegerParameter(200, 600),
        "learning_rate": ContinuousParameter(0.01, 0.1, scaling_type="Logarithmic"),
    },
    # How SageMaker learns the objective: it does NOT read a return value or a
    # file. Each training job's stdout goes to CloudWatch, and SageMaker scrapes it
    # with these regexes, capturing the last match of the (…) group as the metric.
    # So the contract is a print statement: the container must emit a line the regex
    # matches. Here challenger/train.py prints "validation_auc: 0.882431" and this
    # regex captures 0.882431; objective_metric_name must equal one Name below. If
    # the container never prints a matching line, the metric is never captured — the
    # tuner runs but cannot rank trials. (The same metric_definitions on a plain
    # ModelTrainer is also what surfaces training metrics in the console/CloudWatch.)
    metric_definitions=[
        {"Name": "validation_auc", "Regex": "validation_auc: ([0-9\\.]+)"}
    ],
    strategy="Bayesian",
    max_jobs=6,
    max_parallel_jobs=2,
    early_stopping_type="Auto",
)

# -------------------------------------------------------------------------------
# Launch
# -------------------------------------------------------------------------------
tuner.tune(
    inputs=[
        InputData(channel_name="train", data_source=f"s3://{BUCKET}/ch02/input/train/"),
        InputData(
            channel_name="validation",
            data_source=f"s3://{BUCKET}/ch02/input/validation/",
        ),
    ],
    wait=True,
)
print("best training job:", tuner.best_training_job())
