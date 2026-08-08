# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "boto3"]
# ///
"""SageMaker Automatic Model Tuning for the challenger (SDK v3).

Instance-based HPO: a HyperparameterTuner drives training jobs of the BYOC image over ranges and returns the best, every trial captured in the serverless MLflow App. Needs training-job quota (0 on a fresh account, see aws/README); for a quota-free search use src/tuning/amt.py.

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
# ml.m5.large; AMT spends training-job instances, so needs "ml.m5.large for training job usage" quota (fresh account = 0). Cold-run 2026-07-23 (tuning job Completed).
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")

# -------------------------------------------------------------------------------
# The base trainer
# -------------------------------------------------------------------------------
# The training image and its static (non-tuned) settings, exactly as a single
# training job would run; the tuner varies the ranges below on top of this.
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
    # SageMaker scrapes each training job's stdout from CloudWatch with these regexes (last match of the group is the metric), so the container must print a matching line.
    # src/challenger/train.py prints "validation_auc: 0.882431"; objective_metric_name must equal one Name below.
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
