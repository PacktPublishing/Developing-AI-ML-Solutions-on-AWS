.PHONY: iam deploy classify route eval round teardown image prepare decide decider-round

# the whole AWS round in one command (run make iam once first, as the account admin): the
# stack, the batch classification (Bedrock), routing, and the evaluation. make teardown when done.
round:
	$(MAKE) deploy
	$(MAKE) classify
	$(MAKE) route
	$(MAKE) eval

REGION := us-east-1
STACK := ch10-conversation-classification
# The batch model, addressed by its cross-region inference profile. Nova Lite by default:
# its batch pool picks jobs up in minutes, where the in-demand Claude profiles can sit in
# the queue for hours. The manifest is Converse-shaped, so any batch-capable model works
# unchanged -- override BATCH_MODEL with a Claude or other profile if its batch capacity is
# responsive. List them with: aws bedrock list-inference-profiles.
BATCH_MODEL ?= us.amazon.nova-lite-v1:0
DATASET ?= conversations
# The decision model is off by default: it provisions a VPC and GPU EC2 in the
# account, which bills by the hour while a session is alive.
WITH_DECIDER ?= false
# Empty on the first deploy, which creates the repository the image goes to.
DECIDER_IMAGE ?=
# prompt variant: a (bare category names) or b (defined category borders)
VARIANT ?= a

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) --query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)
BUCKET = $(call output,BatchBucket)
ROLE = $(call output,BatchServiceRoleArn)
DECIDER_REPO = $(call output,DeciderRepoUri)
DECIDER_ARN = $(call output,DeciderRuntimeArn)
DECIDER_SRC = $(call output,DeciderSourceBucket)

# The cloud round talks to the real services: no BEDROCK_LOCAL, no S3_ENDPOINT.
CLOUDENV = PYTHONPATH=src BEDROCK_REGION=$(REGION) BATCH_MODEL=$(BATCH_MODEL) \
	BEDROCK_BATCH_ROLE=$(ROLE)

# -------------------------------------------------------------------------------
# deploy -> classify (real batch job) -> route (real SNS) -> eval
# -------------------------------------------------------------------------------
deploy:
	aws cloudformation deploy --region $(REGION) --stack-name $(STACK) \
		--template-file template.yaml --no-fail-on-empty-changeset \
		--capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
		--parameter-overrides WithDecider=$(WITH_DECIDER) DeciderImageUri=$(DECIDER_IMAGE)

# classify: submit a real CreateModelInvocationJob and wait for it (tens of minutes)
classify:
	cd .. && $(CLOUDENV) uv run src/classify.py --dataset $(DATASET) --bucket $(BUCKET) --variant $(VARIANT)

# route: publish each classified conversation to its team's real SNS topic
route:
	cd .. && PYTHONPATH=src BEDROCK_REGION=$(REGION) uv run src/route.py --dataset $(DATASET)

eval:
	cd .. && PYTHONPATH=src uv run src/evaluate.py --dataset $(DATASET)

# -------------------------------------------------------------------------------
# The decision-model round: deploy what the build needs, build the image, deploy
# the runtime against it, put torch and the weights on the volume, then route.
# -------------------------------------------------------------------------------
decider-round:
	$(MAKE) deploy WITH_DECIDER=true
	$(MAKE) image
	$(MAKE) deploy WITH_DECIDER=true DECIDER_IMAGE=$(DECIDER_REPO):latest
	$(MAKE) prepare
	$(MAKE) decide
	$(MAKE) eval

# image: built in the account by CodeBuild, never pushed from a laptop, so the
# x86_64 GPU image does not depend on what this machine can build.
image:
	rm -rf /tmp/ch10-build && mkdir -p /tmp/ch10-build
	cp ../agentcore/Dockerfile ../agentcore/app.py ../agentcore/requirements.txt /tmp/ch10-build/
	cp buildspec.yml /tmp/ch10-build/
	rm -f /tmp/ch10-src.zip && cd /tmp/ch10-build && zip -qr /tmp/ch10-src.zip . -x '*__pycache__*'
	@BUCKET=$$(aws cloudformation describe-stacks --region $(REGION) --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='DeciderSourceBucket'].OutputValue" --output text); \
		aws s3 cp /tmp/ch10-src.zip s3://$$BUCKET/src.zip --region $(REGION); \
		ID=$$(aws codebuild start-build --region $(REGION) --project-name ch10-decider-build \
			--query 'build.id' --output text); \
		echo "build $$ID"; \
		while STATUS=$$(aws codebuild batch-get-builds --region $(REGION) --ids $$ID \
			--query 'builds[0].buildStatus' --output text); \
			[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
			echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

# prepare: the one call that installs torch and pulls the weights onto the
# volume. Minutes, once per volume, and a resumed session skips it.
prepare:
	uv run --no-project invoke.py --region $(REGION) --arn $(DECIDER_ARN) \
		--payload '{"mode": "prepare"}'

# decide: route the dataset through the deployed runtime
decide:
	cd .. && PYTHONPATH=src DECIDER_BACKEND=agentcore AWS_DEFAULT_REGION=$(REGION) \
		AGENT_RUNTIME_ARN=$(DECIDER_ARN) \
		uv run --with boto3 src/decide.py --dataset $(DATASET) --variant b

teardown:
	-aws s3 rm s3://$(BUCKET) --recursive --region $(REGION)
	@SRC=$$(aws cloudformation describe-stacks --region $(REGION) --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='DeciderSourceBucket'].OutputValue" \
			--output text 2>/dev/null); \
		if [ -n "$$SRC" ] && [ "$$SRC" != "None" ]; then \
			echo "emptying $$SRC"; aws s3 rm s3://$$SRC --recursive --region $(REGION) || true; \
		fi
	aws cloudformation delete-stack --region $(REGION) --stack-name $(STACK)
	aws cloudformation wait stack-delete-complete --region $(REGION) --stack-name $(STACK)
	@for t in $$(aws sns list-topics --region $(REGION) --query "Topics[?contains(TopicArn, ':ch10-')].TopicArn" --output text); do \
		echo "deleting topic $$t"; aws sns delete-topic --region $(REGION) --topic-arn $$t; done
	@echo "$(STACK) torn down"

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch10-user --description "Deploy/operate ch-10" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch10-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch10-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch10-user ready"
