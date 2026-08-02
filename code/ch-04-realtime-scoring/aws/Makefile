.PHONY: iam lint push-models endpoints endpoints-teardown deploy deploy-ecs deploy-eks image scale-up scale-down decide decide-ecs decide-eks teardown teardown-ecs teardown-eks

REGION ?= us-east-1
STACK := realtime-scoring-ch04

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) \
	--query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)

lint:
	uvx cfn-lint template.yaml
	rain fmt --verify template.yaml

# --- compute target: ecs (ALB + Fargate service, the default) or eks (an eksctl
# Fargate cluster). Both run the same container image and write to the same
# DynamoDB — the SAM stack supplies both — they differ only in where the pods
# run. Pick the other with COMPUTE=eks. ----------------------------------------
COMPUTE ?= ecs
REPO = $(call output,ServiceRepoUri)

# --- models: the two SageMaker endpoints the gateway ensembles, built and
# deployed apart from the gateway stack. push-models builds each BYOC image
# (linux/amd64, docker v2 manifest — what SageMaker requires) and pushes it;
# endpoints deploys them serverless and needs a SageMaker execution role. -------
ACCOUNT = $(shell aws sts get-caller-identity --query Account --output text)
ECR = $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com
SM_ROLE ?=
MODELS := scorecard challenger

push-models:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ECR)
	@for m in $(MODELS); do \
		aws ecr describe-repositories --repository-names ch04-$$m-model --region $(REGION) >/dev/null 2>&1 || \
		aws ecr create-repository --repository-name ch04-$$m-model --region $(REGION) >/dev/null; \
		docker buildx build --platform linux/amd64 --provenance=false \
		-t $(ECR)/ch04-$$m-model:latest --push ../src/models/$$m; \
	done

endpoints:
	@test -n "$(SM_ROLE)" || { echo "set SM_ROLE=<sagemaker exec role arn> (make iam prints it)"; exit 1; }
	@for m in $(MODELS); do \
		MODE=cloud MODEL_IMAGE=$(ECR)/ch04-$$m-model:latest SAGEMAKER_ROLE_ARN=$(SM_ROLE) \
		ENDPOINT_NAME=ch04-$$m uv run --no-project ../src/deploy.py; \
	done

endpoints-teardown:
	@for m in $(MODELS); do \
		aws sagemaker delete-endpoint --endpoint-name ch04-$$m --region $(REGION) 2>/dev/null || true; \
		aws sagemaker delete-endpoint-config --endpoint-config-name ch04-$$m --region $(REGION) 2>/dev/null || true; \
		aws sagemaker delete-model --model-name ch04-$$m --region $(REGION) 2>/dev/null || true; \
	done

deploy: deploy-$(COMPUTE)
decide: decide-$(COMPUTE)
teardown: endpoints-teardown teardown-$(COMPUTE)

# ECS: the SAM stack runs the service behind the ALB (deploy, build, scale to 1).
deploy-ecs:
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset
	$(MAKE) image
	$(MAKE) scale-up

# EKS: the SAM stack still supplies the ECR image and the DynamoDB table; the pods
# run on an eksctl Fargate cluster (its own VPC) instead of ECS.
deploy-eks:
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset
	$(MAKE) image
	eksctl create cluster -f eks/create-cluster.yaml
	sed -e "s|ch04-decision:local|$(REPO):latest|" \
		-e "s|imagePullPolicy: Never|imagePullPolicy: Always|" \
		../k8s/gateway.yaml | kubectl apply -f -
	kubectl rollout status deployment/gateway --timeout=300s

# build the gateway image in the cloud and push it to the stack's ECR repo
image:
	$(eval BUCKET := $(call output,StagingBucketName))
	cd .. && rm -f /tmp/ch04-app-src.zip && \
		zip -q -j /tmp/ch04-app-src.zip src/gateway/Dockerfile src/gateway/app.py \
		aws/image/buildspec.yml
	aws s3 cp /tmp/ch04-app-src.zip s3://$(BUCKET)/ch04/source/app.zip
	@ID=$$(aws codebuild start-build --region $(REGION) --project-name ch04-decision-build \
		--query 'build.id' --output text); \
	echo "build $$ID"; \
	while STATUS=$$(aws codebuild batch-get-builds --region $(REGION) --ids $$ID \
		--query 'builds[0].buildStatus' --output text); \
		[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
		echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

# start one task behind the ALB (the stack deploys at desired count 0)
scale-up:
	aws ecs update-service --region $(REGION) --cluster ch04-decision \
		--service decision-service --desired-count 1 >/dev/null
	aws ecs wait services-stable --region $(REGION) --cluster ch04-decision \
		--services decision-service
	@echo "service up: $(call output,ServiceUrl)"

scale-down:
	aws ecs update-service --region $(REGION) --cluster ch04-decision \
		--service decision-service --desired-count 0 >/dev/null

# one application through the real endpoint: the ALB for ecs, a port-forward for eks
DECIDE_BODY = '{"application_id":"app-1042","age":40,"monthly_income":6000,"requested_amount":5000,"dti":0.10,"utilization":0.10}'
decide-ecs:
	curl -s $(call output,ServiceUrl)/decision -H 'content-type: application/json' -d $(DECIDE_BODY)
	@echo

decide-eks:
	@kubectl port-forward svc/gateway 8085:80 >/dev/null 2>&1 & \
	PF=$$!; sleep 3; \
	curl -s localhost:8085/decision -H 'content-type: application/json' -d $(DECIDE_BODY); \
	echo; kill $$PF

teardown-ecs:
	-aws ecs update-service --region $(REGION) --cluster ch04-decision \
		--service decision-service --desired-count 0
	-aws s3 rm --recursive s3://$(call output,StagingBucketName)
	sam delete --no-prompts --region $(REGION)

teardown-eks:
	-eksctl delete cluster -f eks/create-cluster.yaml
	-aws s3 rm --recursive s3://$(call output,StagingBucketName)
	sam delete --no-prompts --region $(REGION)

# --- iam: provision THIS chapter's deploy identity, when you start the chapter -
# Creates ch04-user from iam/deploy.json (least-privilege for ch-04 only) and trusts you to
# assume it — per chapter, not an account-wide up-front bootstrap. Re-run after
# editing the policy. Then add a [profile ch04] (role_arn + source_profile) and
# run cloud targets with AWS_PROFILE=ch04.
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch04-user --description "Deploy/operate ch-04" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch04-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch04-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch04-user ready"
	@aws iam create-role --role-name ch04-sagemaker --description "SageMaker endpoint execution role for ch-04" \
		--assume-role-policy-document file://iam/sagemaker-trust.json 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch04-sagemaker --policy-document file://iam/sagemaker-trust.json
	aws iam put-role-policy --role-name ch04-sagemaker --policy-name serve --policy-document file://iam/sagemaker-policy.json
	@echo "ch04-sagemaker ready — pass its ARN as SM_ROLE to make endpoints / sm-local:"
	@aws iam get-role --role-name ch04-sagemaker --query Role.Arn --output text