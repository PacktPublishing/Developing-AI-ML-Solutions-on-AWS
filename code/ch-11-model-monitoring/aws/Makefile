.PHONY: iam deploy image pipeline teardown

# Reproduce ch-11's monitoring on real AWS as a SageMaker Pipeline: Batch Transform
# (score the current batch) -> Processing (PSI + SHAP feature-attribution drift, AWS's
# documented replacement for the closed Clarify service) + CloudWatch. Account values
# are derived; override on the command line.
REGION ?= us-east-1
ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text)
REGISTRY := $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com
IMAGE := $(REGISTRY)/ch11-scorecard:latest
STACK := ch11-monitoring

# the SageMaker default bucket (created if missing) and the SageMaker execution role
BUCKET ?= sagemaker-$(REGION)-$(ACCOUNT)
ROLE ?= $(shell aws iam list-roles --query "Roles[?contains(RoleName,'AmazonSageMaker-ExecutionRole')].Arn | [0]" --output text)

# -------------------------------------------------------------------------------
# iam: provision this chapter's deploy identity (run once, as the account admin)
# -------------------------------------------------------------------------------
# Creates ch11-user from iam/deploy.json (least-privilege for ch-11) and trusts you to
# assume it; then add a [profile ch11] and run the cloud targets with AWS_PROFILE=ch11.
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch11-user --description "Deploy/operate ch-11" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch11-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch11-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch11-user ready: arn:aws:iam::$(ACCOUNT):role/ch11-user"

# -------------------------------------------------------------------------------
# deploy: the ECR repo + the feature-attribution-drift alarm (before make image)
# -------------------------------------------------------------------------------
deploy:
	aws cloudformation deploy --template-file template.yaml --stack-name $(STACK)
	@echo "stack $(STACK) ready"

# -------------------------------------------------------------------------------
# image: build the train/serve/monitor image for linux/amd64 and push to ECR
# -------------------------------------------------------------------------------
image:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(REGISTRY)
	docker buildx build --platform linux/amd64 --provenance=false \
		--output type=image,name=$(IMAGE),oci-mediatypes=false,push=true \
		-f ../Dockerfile ..
	@echo "pushed $(IMAGE)"

# -------------------------------------------------------------------------------
# pipeline: Batch Transform -> monitoring Processing step, then the alarms
# -------------------------------------------------------------------------------
# Train the scorecard locally first (make data && make train in the chapter root):
# the pipeline packages that model, scores the current batch, and monitors the drift.
pipeline:
	@aws s3api head-bucket --bucket $(BUCKET) 2>/dev/null \
		|| aws s3 mb s3://$(BUCKET) --region $(REGION)
	IMAGE_URI=$(IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		AWS_DEFAULT_REGION=$(REGION) uv run pipeline.py

teardown:
	-aws s3 rm s3://$(BUCKET)/ch11 --recursive
	-aws sagemaker delete-model --model-name ch11-scorecard-transform 2>/dev/null
	-aws sagemaker delete-pipeline --pipeline-name ch11-monitoring 2>/dev/null
	aws cloudformation delete-stack --stack-name $(STACK)
	aws cloudformation wait stack-delete-complete --stack-name $(STACK)
	@echo "$(STACK) torn down"