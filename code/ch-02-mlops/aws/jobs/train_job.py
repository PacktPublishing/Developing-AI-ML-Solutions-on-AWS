# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sagemaker>=3,<4", "boto3"]
# ///
"""Run the scorecard container as a real SageMaker training job (SDK v3).

Parity with local mode: the container is byte-identical (the image CodeBuild
pushed to ECR). Only the runner changes — local mode does `docker run ... train`,
this submits the same image to a managed training job through the v3
sagemaker.train.ModelTrainer. Set MLFLOW_TRACKING_URI to a serverless MLflow App
ARN to log the run; leave it unset to just produce model.tar.gz in S3.

Env:
  IMAGE_URI, SAGEMAKER_ROLE_ARN, ARTIFACT_BUCKET (required)
  AWS_DEFAULT_REGION, INSTANCE_TYPE, MLFLOW_TRACKING_URI (optional)
"""

import os

import boto3
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (
    Compute,
    InputData,
    OutputDataConfig,
    StoppingCondition,
)

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
IMAGE_URI = os.environ["IMAGE_URI"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")

env = {
    k: v
    for k, v in {
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
        "MLFLOW_ARTIFACT_ROOT": os.environ.get("MLFLOW_ARTIFACT_ROOT", ""),
    }.items()
    if v
}

trainer = ModelTrainer(
    training_image=IMAGE_URI,
    role=ROLE,
    base_job_name="ch02-scorecard",
    compute=Compute(
        instance_type=INSTANCE_TYPE, instance_count=1, volume_size_in_gb=30
    ),
    output_data_config=OutputDataConfig(s3_output_path=f"s3://{BUCKET}/ch02/output"),
    hyperparameters={"C": "1.0", "max_iter": "1000", "monotonic": "true"},
    environment=env or None,
    stopping_condition=StoppingCondition(max_runtime_in_seconds=1800),
)

trainer.train(
    input_data_config=[
        InputData(channel_name="train", data_source=f"s3://{BUCKET}/ch02/input/train/"),
        InputData(
            channel_name="validation",
            data_source=f"s3://{BUCKET}/ch02/input/validation/",
        ),
    ],
    wait=True,
    logs=True,
)

# Report the model artifact for the deploy step.
sm = boto3.client("sagemaker", region_name=REGION)
jobs = sm.list_training_jobs(
    NameContains="ch02-scorecard",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)
job_name = jobs["TrainingJobSummaries"][0]["TrainingJobName"]
desc = sm.describe_training_job(TrainingJobName=job_name)
print("job:", job_name, "status:", desc["TrainingJobStatus"])
print("model_data:", desc["ModelArtifacts"]["S3ModelArtifacts"])
