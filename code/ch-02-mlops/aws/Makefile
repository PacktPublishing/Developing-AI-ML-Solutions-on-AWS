.PHONY: image mlflow-app mlflow-app-status mlflow-app-delete train amt \
	deploy-byoc deploy-modelbuilder

# Reproduce the chapter on real AWS with SDK v3. Account-specific values are
# derived where possible; override any of them on the command line, e.g.
#   make train ROLE=arn:aws:iam::...:role/MySageMakerRole
REGION ?= us-east-1
ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text)
REGISTRY := $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com
CHALLENGER_IMAGE := $(REGISTRY)/ch02-challenger:latest

# the SageMaker default bucket, an execution role, and the MLflow App ARN
BUCKET ?= $(shell aws s3api list-buckets --query "Buckets[?starts_with(Name,'amazon-sagemaker-$(ACCOUNT)')].Name | [0]" --output text)
ROLE ?= $(shell aws iam list-roles --query "Roles[?contains(RoleName,'AmazonSageMaker-ExecutionRole')].Arn | [0]" --output text)
APP_ARN ?= $(shell aws sagemaker list-mlflow-apps --region $(REGION) --query "Summaries[?Name=='ch02-mlflow'].Arn | [0]" --output text)

MODEL_PATH ?= models:/credit-challenger/1

# --- image: build the container in the cloud (CodeBuild -> ECR) ---------------
# image/buildspec.yml is the build; image/github-actions.yml triggers it on push.
# Assumes the CodeBuild projects exist (see README for one-time setup).
image:
	aws codebuild start-build --project-name ch02-scorecard-build
	aws codebuild start-build --project-name ch02-challenger-build

# --- mlflow app: the serverless tracking server's successor -------------------
mlflow-app:
	MLFLOW_APP_ARTIFACT_URI=s3://$(BUCKET)/ch02-mlflow SAGEMAKER_ROLE_ARN=$(ROLE) \
		AWS_DEFAULT_REGION=$(REGION) sh mlflow_app.sh create
mlflow-app-status:
	SAGEMAKER_ROLE_ARN=$(ROLE) MLFLOW_APP_ARTIFACT_URI=s3://$(BUCKET)/ch02-mlflow \
		AWS_DEFAULT_REGION=$(REGION) sh mlflow_app.sh status
mlflow-app-delete:
	SAGEMAKER_ROLE_ARN=$(ROLE) MLFLOW_APP_ARTIFACT_URI=s3://$(BUCKET)/ch02-mlflow \
		AWS_DEFAULT_REGION=$(REGION) sh mlflow_app.sh delete

# --- jobs: training and tuning (need training-instance quota) -----------------
train:
	IMAGE_URI=$(CHALLENGER_IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		MLFLOW_TRACKING_URI=$(APP_ARN) AWS_DEFAULT_REGION=$(REGION) uv run jobs/train_job.py
amt:
	IMAGE_URI=$(CHALLENGER_IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		MLFLOW_TRACKING_ARN=$(APP_ARN) AWS_DEFAULT_REGION=$(REGION) uv run jobs/amt.py

# --- deploy: a registered model, two ways -------------------------------------
deploy-byoc:
	IMAGE_URI=$(CHALLENGER_IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		MLFLOW_TRACKING_ARN=$(APP_ARN) MLFLOW_MODEL_PATH=$(MODEL_PATH) \
		AWS_DEFAULT_REGION=$(REGION) uv run deploy/deploy_byoc_from_registry.py
deploy-modelbuilder:
	SAGEMAKER_ROLE_ARN=$(ROLE) MLFLOW_TRACKING_ARN=$(APP_ARN) MLFLOW_MODEL_PATH=$(MODEL_PATH) \
		AWS_DEFAULT_REGION=$(REGION) uv run deploy/deploy_modelbuilder.py