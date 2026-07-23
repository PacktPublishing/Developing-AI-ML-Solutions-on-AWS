#!/bin/sh
# Create / describe / delete the serverless MLflow App (SageMaker AI "MLflow
# Apps", GA Dec 2025). This is the free, scale-to-zero successor to the always-on
# MLflow tracking server: no server to size, tracking is there when a job logs to
# it. The App's ARN is the MLFLOW_TRACKING_URI clients use (with the
# sagemaker-mlflow plugin) — the one thing that differs from local mode, where
# the same variable is a sqlite path.
#
# Defined here as a CLI command rather than CloudFormation because the
# AWS::SageMaker::MlflowApp resource type is new; move this to the aws/ SAM
# template once the resource is available in your region's CloudFormation.
#
# Usage: sh aws/mlflow_app.sh create | status | uri | delete
set -e

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
NAME="${MLFLOW_APP_NAME:-ch02-mlflow}"
ARTIFACT_URI="${MLFLOW_APP_ARTIFACT_URI:?set MLFLOW_APP_ARTIFACT_URI=s3://your-bucket/ch02-mlflow}"
ROLE_ARN="${SAGEMAKER_ROLE_ARN:?set SAGEMAKER_ROLE_ARN}"

app_arn() {
  aws sagemaker list-mlflow-apps --region "$REGION" \
    --query "Summaries[?Name=='$NAME'].Arn | [0]" --output text
}

case "$1" in
create)
  aws sagemaker create-mlflow-app --region "$REGION" \
    --name "$NAME" \
    --artifact-store-uri "$ARTIFACT_URI" \
    --role-arn "$ROLE_ARN" \
    --model-registration-mode AutoModelRegistrationEnabled
  ;;
status)
  aws sagemaker describe-mlflow-app --region "$REGION" --arn "$(app_arn)" \
    --query "{Name:Name,Status:Status,Arn:Arn}" --output table
  ;;
uri)
  # the ARN is the MLFLOW_TRACKING_URI (with the sagemaker-mlflow plugin)
  app_arn
  ;;
delete)
  aws sagemaker delete-mlflow-app --region "$REGION" --arn "$(app_arn)"
  ;;
*)
  echo "usage: sh aws/mlflow_app.sh create | status | uri | delete" >&2
  exit 1
  ;;
esac
