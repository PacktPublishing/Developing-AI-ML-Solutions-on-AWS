.PHONY: iam deploy image pipeline clarify endpoint round teardown

# the whole AWS round in one command (run make iam once first, as the account admin): the
# stack, the image built by CodeBuild, then the monitoring pipeline. make teardown when done.
round:
	$(MAKE) deploy
	$(MAKE) image
	$(MAKE) pipeline

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
	aws cloudformation deploy --template-file template.yaml --stack-name $(STACK) \
		--capabilities CAPABILITY_NAMED_IAM
	@echo "stack $(STACK) ready"

# -------------------------------------------------------------------------------
# image: build the train/serve/monitor image in-account with CodeBuild and push to ECR
# -------------------------------------------------------------------------------
# Stage the build context (Dockerfile, src/, entrypoint.sh) at the zip root beside the
# buildspec, upload it to the stack's SourceBucket, start the build, and poll to SUCCEEDED.
image:
	rm -rf /tmp/ch11-build && mkdir -p /tmp/ch11-build
	cp -R ../Dockerfile ../entrypoint.sh ../src /tmp/ch11-build/
	cp buildspec.yml /tmp/ch11-build/
	rm -f /tmp/ch11-src.zip && cd /tmp/ch11-build && zip -qr /tmp/ch11-src.zip . -x '*__pycache__*'
	@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text); \
		aws s3 cp /tmp/ch11-src.zip s3://$$BUCKET/src.zip; \
		ID=$$(aws codebuild start-build --project-name ch11-scorecard-build --query 'build.id' --output text); \
		echo "build $$ID"; \
		while STATUS=$$(aws codebuild batch-get-builds --ids $$ID --query 'builds[0].buildStatus' --output text); \
			[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
			echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]
	@echo "pushed $(IMAGE)"

# -------------------------------------------------------------------------------
# pipeline: Batch Transform -> monitoring Processing step, then the alarms
# -------------------------------------------------------------------------------
# Train the scorecard locally first (make data && make train in the chapter root):
# the pipeline packages that model, scores the current batch, and monitors the drift.
pipeline:
	@aws s3api head-bucket --bucket $(BUCKET) 2>/dev/null \
		|| aws s3 mb s3://$(BUCKET) --region $(REGION)
	PIPELINE_MODE=aws IMAGE_URI=$(IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		AWS_DEFAULT_REGION=$(REGION) uv run pipeline.py

# -------------------------------------------------------------------------------
# clarify: verify the from-source SHAP drift against REAL SageMaker Clarify
# -------------------------------------------------------------------------------
# Optional parity check. Runs two Clarify explainability jobs (KernelSHAP) and compares
# the feature-attribution NDCG with the chapter's from-source (TreeSHAP) NDCG. Clarify is
# closed to new accounts, so this needs a grandfathered account; make pipeline is the
# everyday path. Needs make deploy + make image first (for the scorecard serving image).
clarify:
	@aws s3api head-bucket --bucket $(BUCKET) 2>/dev/null \
		|| aws s3 mb s3://$(BUCKET) --region $(REGION)
	IMAGE_URI=$(IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		AWS_DEFAULT_REGION=$(REGION) PYTHONPATH=$(CURDIR)/../src uv run clarify_monitor.py

# -------------------------------------------------------------------------------
# endpoint: real-time endpoint with Data Capture, then monitor the captured traffic
# -------------------------------------------------------------------------------
# The real-time counterpart to make pipeline: deploys the scorecard image as an endpoint
# with DataCaptureConfig on, sends the current batch through, and prints the capture S3
# prefix to feed run_monitor.py --capture. Needs make image first (for the serving image).
endpoint:
	@aws s3api head-bucket --bucket $(BUCKET) 2>/dev/null \
		|| aws s3 mb s3://$(BUCKET) --region $(REGION)
	IMAGE_URI=$(IMAGE) SAGEMAKER_ROLE_ARN=$(ROLE) ARTIFACT_BUCKET=$(BUCKET) \
		AWS_DEFAULT_REGION=$(REGION) PYTHONPATH=$(CURDIR)/../src uv run endpoint.py

teardown:
	-aws s3 rm s3://$(BUCKET)/ch11 --recursive
	-aws sagemaker delete-endpoint --endpoint-name ch11-scorecard 2>/dev/null
	-@for c in $$(aws sagemaker list-endpoint-configs --name-contains ch11-scorecard \
		--query 'EndpointConfigs[].EndpointConfigName' --output text 2>/dev/null); do \
		aws sagemaker delete-endpoint-config --endpoint-config-name $$c; done
	-@for m in $$(aws sagemaker list-models --name-contains ch11-scorecard \
		--query 'Models[].ModelName' --output text 2>/dev/null); do \
		aws sagemaker delete-model --model-name $$m; done
	-aws sagemaker delete-pipeline --pipeline-name ch11-monitoring 2>/dev/null
	-@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text 2>/dev/null); \
		if [ -n "$$BUCKET" ] && [ "$$BUCKET" != "None" ]; then \
			echo "emptying $$BUCKET"; aws s3 rm s3://$$BUCKET --recursive; fi
	aws cloudformation delete-stack --stack-name $(STACK)
	aws cloudformation wait stack-delete-complete --stack-name $(STACK)
	@echo "$(STACK) torn down"