#!/bin/sh
# Create / describe / delete the serverless MLflow App (SageMaker AI "MLflow Apps", GA Dec 2025), the scale-to-zero successor to the tracking server; its ARN is the MLFLOW_TRACKING_URI clients use.
#
# Defined as a CLI command because AWS::SageMaker::MlflowApp is new; move to the aws/ SAM template once it's in CloudFormation.
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
