.PHONY: iam lint ecr-login push-models endpoints endpoints-teardown deploy deploy-ecs deploy-eks image scale-up scale-down decide decide-ecs decide-eks teardown teardown-ecs teardown-eks

# single-region chapter: the EKS config and gateway manifest hardcode us-east-1, so don't override this for COMPUTE=eks.
REGION := us-east-1
STACK := realtime-scoring-ch04

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) \
	--query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)

lint:
	uvx cfn-lint template.yaml
	rain fmt --verify template.yaml

# -------------------------------------------------------------------------------
# compute target: ecs (ALB + Fargate, the default) or eks (an eksctl Fargate cluster)
# -------------------------------------------------------------------------------
COMPUTE ?= ecs

# -------------------------------------------------------------------------------
# models: the two SageMaker endpoints the gateway ensembles, deployed apart from the stack
# -------------------------------------------------------------------------------
ACCOUNT = $(shell aws sts get-caller-identity --query Account --output text)
ECR = $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com
SM_ROLE ?=
MODELS := scorecard challenger

# One target per model, so the two are independent prerequisites: run them in
# parallel with make -j2 (build/deploy/teardown both models at once) or serially
# with plain make. push-model-% builds and pushes; endpoint-% / teardown-endpoint-%
# deploy and remove one serverless endpoint. endpoint-% depends on its own
# push-model-% so -j2 overlaps the two models without deploying an unpushed tag;
# that means make endpoints always rebuilds+repushes (pattern targets have no file,
# so they are never up to date) -- skip with make -o push-model-<m> endpoint-<m>.
ecr-login:
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ECR)

push-model-%: ecr-login
	aws ecr describe-repositories --repository-names ch04-$*-model --region $(REGION) >/dev/null 2>&1 || \
		aws ecr create-repository --repository-name ch04-$*-model --region $(REGION) >/dev/null
	docker buildx build --platform linux/amd64 --provenance=false \
		--output type=image,oci-mediatypes=false,push=true \
		-t $(ECR)/ch04-$*-model:latest ../src/models/$*

push-models: $(MODELS:%=push-model-%)

# Fail at parse time if a deploy is requested without SM_ROLE: endpoint-% now runs
# after push-model-%, so a recipe-line check would only fire after two image pushes.
ifneq (,$(filter endpoints $(MODELS:%=endpoint-%),$(MAKECMDGOALS)))
ifeq (,$(SM_ROLE))
$(error set SM_ROLE=<sagemaker exec role arn> (make iam prints it))
endif
endif

endpoint-%: push-model-%
	MODE=cloud MODEL_IMAGE=$(ECR)/ch04-$*-model:latest SAGEMAKER_ROLE_ARN=$(SM_ROLE) \
		ENDPOINT_NAME=ch04-$* uv run --no-project ../src/deploy.py

endpoints: $(MODELS:%=endpoint-%)

# teardown-endpoint-% (not endpoint-teardown-%), which would also match the
# endpoint-% pattern above. GNU make >= 3.82 would pick the shortest stem and
# get it right; make 3.81 (still /usr/bin/make on macOS) picks by file order
# and would run the deploy recipe instead. This naming is version-independent.
teardown-endpoint-%:
	-aws sagemaker delete-endpoint --endpoint-name ch04-$* --region $(REGION) 2>/dev/null
	-aws sagemaker delete-endpoint-config --endpoint-config-name ch04-$* --region $(REGION) 2>/dev/null
	-aws sagemaker delete-model --model-name ch04-$* --region $(REGION) 2>/dev/null

endpoints-teardown: $(MODELS:%=teardown-endpoint-%)

deploy: deploy-$(COMPUTE)
decide: decide-$(COMPUTE)
teardown: endpoints-teardown teardown-$(COMPUTE)

# ECS: the SAM stack builds the Fargate runtime behind the ALB (deploy, build, scale to 1).
deploy-ecs:
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset --parameter-overrides Compute=ecs
	$(MAKE) image
	$(MAKE) scale-up

# EKS: with Compute=eks the stack builds only the ECR repo, DynamoDB table, and image build; the gateway pod runs on an eksctl Fargate cluster.
deploy-eks:
	@command -v eksctl >/dev/null 2>&1 || { echo "eksctl is not installed: https://eksctl.io/installation/"; exit 1; }
	@command -v kubectl >/dev/null 2>&1 || { echo "kubectl is not installed: https://kubernetes.io/docs/tasks/tools/"; exit 1; }
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset --parameter-overrides Compute=eks
	$(MAKE) image
	eksctl create cluster -f eks/create-cluster.yaml
	REPO_URI=$$(aws cloudformation describe-stacks --region $(REGION) --stack-name $(STACK) \
		--query "Stacks[0].Outputs[?OutputKey=='ServiceRepoUri'].OutputValue" --output text); \
		test -n "$$REPO_URI" || { echo "ServiceRepoUri output is empty; is the stack up?"; exit 1; }; \
		sed -e "s|ch04-decision:local|$${REPO_URI}:latest|" \
		-e "s|imagePullPolicy: Never|imagePullPolicy: Always|" \
		../k8s/gateway.yaml | kubectl apply -f -
	kubectl rollout status deployment/gateway --timeout=300s

# the staging bucket, resolved once at point of use: a lazy global, not a
# recipe-time $(eval) -- eval mutates global make state during expansion, the one
# construct that genuinely misbehaves under make -j.
BUCKET = $(call output,StagingBucketName)

# build the gateway image in the cloud and push it to the stack's ECR repo
image:
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
	@command -v kubectl >/dev/null 2>&1 || { echo "kubectl is not installed"; exit 1; }
	uv run --no-project eks/decide.py

teardown-ecs:
	-aws ecs update-service --region $(REGION) --cluster ch04-decision \
		--service decision-service --desired-count 0
	-aws s3 rm --recursive s3://$(BUCKET)
	sam delete --no-prompts --region $(REGION)

teardown-eks:
	@command -v eksctl >/dev/null 2>&1 || { echo "eksctl is not installed: https://eksctl.io/installation/"; exit 1; }
	-eksctl delete cluster -f eks/create-cluster.yaml
	-aws s3 rm --recursive s3://$(BUCKET)
	sam delete --no-prompts --region $(REGION)

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# -------------------------------------------------------------------------------
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
	@echo "ch04-sagemaker ready; pass its ARN as SM_ROLE to make endpoints / sm-local:"
	@aws iam get-role --role-name ch04-sagemaker --query Role.Arn --output text