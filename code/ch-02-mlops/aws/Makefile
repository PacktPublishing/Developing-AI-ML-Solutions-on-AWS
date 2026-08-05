.PHONY: iam image image-bootstrap image-teardown mlflow-app mlflow-app-status mlflow-app-delete train amt \
	deploy-byoc deploy-modelbuilder

# Reproduce the chapter on real AWS with SDK v3. Account-specific values are
# derived where possible; override any of them on the command line, e.g.
#   make train ROLE=arn:aws:iam::...:role/MySageMakerRole
REGION ?= us-east-1
ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text)
REGISTRY := $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com
SCORECARD_IMAGE := $(REGISTRY)/ch02-scorecard:latest
CHALLENGER_IMAGE := $(REGISTRY)/ch02-challenger:latest

# the SageMaker default bucket, an execution role, and the MLflow App ARN
BUCKET ?= $(shell aws s3api list-buckets --query "Buckets[?starts_with(Name,'amazon-sagemaker-$(ACCOUNT)')].Name | [0]" --output text)
ROLE ?= $(shell aws iam list-roles --query "Roles[?contains(RoleName,'AmazonSageMaker-ExecutionRole')].Arn | [0]" --output text)
APP_ARN ?= $(shell aws sagemaker list-mlflow-apps --region $(REGION) --query "Summaries[?Name=='ch02-mlflow'].Arn | [0]" --output text)

# empty by default: the deploy scripts resolve the latest registered version.
# Pin one to override, e.g. make deploy-byoc MODEL_PATH=models:/credit-challenger/2
MODEL_PATH ?=

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# -------------------------------------------------------------------------------
# Creates ch02-user from iam/deploy.json (least-privilege for ch-02 only) and trusts you to assume it; re-run after editing deploy.json.
# Then add a [profile ch02] (role_arn + source_profile) and run cloud targets AWS_PROFILE=ch02.
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch02-user --description "Deploy/operate ch-02" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch02-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch02-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch02-user ready: arn:aws:iam::$(ACCOUNT):role/ch02-user"

# -------------------------------------------------------------------------------
# image: build the container in the cloud (CodeBuild -> ECR)
# -------------------------------------------------------------------------------
# image-bootstrap creates the ECR repo + CodeBuild project + source bucket (image/build.yaml, once per image); image then zips the context, uploads it, and runs the build.
# REPO=ch02-challenger (train_job.py's image) or ch02-scorecard; image/github-actions.yml is the CI that triggers the same build.
REPO ?= ch02-challenger
SRCDIR = $(if $(filter ch02-scorecard,$(REPO)),scorecard,challenger)

image-bootstrap:
	aws cloudformation deploy --template-file image/build.yaml \
		--stack-name $(REPO)-build --capabilities CAPABILITY_IAM \
		--parameter-overrides RepoName=$(REPO)

# S3 blocks CloudFormation from deleting a non-empty bucket, so empty the source
# bucket before deleting the stack (EmptyOnDelete clears the ECR image).
image-teardown:
	@BKT=$$(aws cloudformation describe-stack-resources --stack-name $(REPO)-build \
		--query "StackResources[?ResourceType=='AWS::S3::Bucket'].PhysicalResourceId" --output text 2>/dev/null); \
	if [ -n "$$BKT" ] && [ "$$BKT" != "None" ]; then echo "emptying s3://$$BKT"; aws s3 rm "s3://$$BKT" --recursive; fi
	aws cloudformation delete-stack --stack-name $(REPO)-build
	aws cloudformation wait stack-delete-complete --stack-name $(REPO)-build
	@echo "$(REPO)-build torn down"

image:
	$(eval BUCKET := $(shell aws cloudformation describe-stacks --stack-name $(REPO)-build \
		--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text))
	rm -f /tmp/$(REPO)-src.zip
	cd ../src/$(SRCDIR) && zip -q -r /tmp/$(REPO)-src.zip . -x '__pycache__/*'
	zip -qj /tmp/$(REPO)-src.zip image/buildspec.yml
	aws s3 cp /tmp/$(REPO)-src.zip s3://$(BUCKET)/src.zip
	@ID=$$(aws codebuild start-build --project-name $(REPO)-build --query 'build.id' --output text); \
	echo "build $$ID"; \
	while STATUS=$$(aws codebuild batch-get-builds --ids $$ID \
		--query 'builds[0].buildStatus' --output text); \
		[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
		echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

# -------------------------------------------------------------------------------
# mlflow app: the serverless tracking server's successor
# -------------------------------------------------------------------------------
mlflow-app:
	MLFLOW_APP_ARTIFACT_URI=s3://$(BUCKET)/ch02-mlflow SAGEMAKER_ROLE_ARN=$(ROLE) \
		AWS_DEFAULT_REGION=$(REGION) sh mlflow_app.sh create
mlflow-app-status:
	SAGEMAKER_ROLE_ARN=$(ROLE) MLFLOW_APP_ARTIFACT_URI=s3://$(BUCKET)/ch02-mlflow \
		AWS_DEFAULT_REGION=$(REGION) sh mlflow_app.sh status
mlflow-app-delete:
	SAGEMAKER_ROLE_ARN=$(ROLE) MLFLOW_APP_ARTIFACT_URI=s3://$(BUCKET)/ch02-mlflow \
		AWS_DEFAULT_REGION=$(REGION) sh mlflow_app.sh delete

# -------------------------------------------------------------------------------
# jobs: training and tuning (need training-instance quota)
# -------------------------------------------------------------------------------
train:
	IMAGE_URI=$(SCORECARD_IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		MLFLOW_TRACKING_URI=$(APP_ARN) AWS_DEFAULT_REGION=$(REGION) uv run jobs/train_job.py
amt:
	IMAGE_URI=$(CHALLENGER_IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		MLFLOW_TRACKING_ARN=$(APP_ARN) AWS_DEFAULT_REGION=$(REGION) uv run jobs/amt.py

# -------------------------------------------------------------------------------
# deploy: a registered model, two ways
# -------------------------------------------------------------------------------
deploy-byoc:
	IMAGE_URI=$(CHALLENGER_IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		MLFLOW_TRACKING_ARN=$(APP_ARN) MLFLOW_MODEL_PATH=$(MODEL_PATH) \
		AWS_DEFAULT_REGION=$(REGION) uv run deploy/deploy_byoc_from_registry.py
deploy-modelbuilder:
	SAGEMAKER_ROLE_ARN=$(ROLE) MLFLOW_TRACKING_ARN=$(APP_ARN) MLFLOW_MODEL_PATH=$(MODEL_PATH) \
		AWS_DEFAULT_REGION=$(REGION) uv run deploy/deploy_modelbuilder.py