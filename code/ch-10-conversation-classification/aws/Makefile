.PHONY: iam deploy classify route eval round teardown image agent decide decider-round

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
# Empty on the first deploy, which creates the repository and the bucket the
# image and the agent zip go to.
DECIDER_IMAGE ?=
AGENT_KEY ?=
# prompt variant: a (bare category names) or b (defined category borders)
VARIANT ?= a

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) --query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)
BUCKET = $(call output,BatchBucket)
ROLE = $(call output,BatchServiceRoleArn)
DECIDER_REPO = $(call output,DeciderRepoUri)
DECIDER_ARN = $(call output,RoutingAgentArn)
DECIDER_SRC = $(call output,DeciderSourceBucket)

# The cloud round talks to the real services: no BEDROCK_LOCAL, no S3_ENDPOINT.
CLOUDENV = PYTHONPATH=src BEDROCK_REGION=$(REGION) BATCH_MODEL=$(BATCH_MODEL) \
	BEDROCK_BATCH_ROLE=$(ROLE)

# -------------------------------------------------------------------------------
# deploy -> classify (real batch job) -> route (real SNS) -> eval
# -------------------------------------------------------------------------------
# A create that fails leaves the stack in ROLLBACK_COMPLETE, and a changeset
# that never executes leaves it in REVIEW_IN_PROGRESS. Neither can be updated,
# and neither holds any resource, so the only way forward is to delete first.
deploy:
	@STATUS=$$(aws cloudformation describe-stacks --region $(REGION) --stack-name $(STACK) \
			--query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo NONE); \
		case "$$STATUS" in \
			ROLLBACK_COMPLETE|REVIEW_IN_PROGRESS) \
			echo "stack is $$STATUS and cannot be updated; deleting it first"; \
			aws cloudformation delete-stack --region $(REGION) --stack-name $(STACK); \
			aws cloudformation wait stack-delete-complete --region $(REGION) --stack-name $(STACK);; \
		esac
	aws cloudformation deploy --region $(REGION) --stack-name $(STACK) \
		--template-file template.yaml --no-fail-on-empty-changeset \
		--capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
		--parameter-overrides WithDecider=$(WITH_DECIDER) \
		DeciderImageUri=$(DECIDER_IMAGE) AgentCodeKey=$(AGENT_KEY)

# classify: submit a real CreateModelInvocationJob and wait for it (tens of minutes)
classify:
	cd .. && $(CLOUDENV) uv run --with boto3 --with "botocore[crt]" src/classify.py --dataset $(DATASET) --bucket $(BUCKET) --variant $(VARIANT)

# route: publish each classified conversation to its team's real SNS topic
route:
	cd .. && PYTHONPATH=src BEDROCK_REGION=$(REGION) uv run --with boto3 --with "botocore[crt]" src/route.py --dataset $(DATASET)

eval:
	cd .. && PYTHONPATH=src uv run src/evaluate.py --dataset $(DATASET)

# -------------------------------------------------------------------------------
# The decision-model round. Two deployables: the model's image, built in the
# account, and the agent's zip, which needs no image at all.
# -------------------------------------------------------------------------------
decider-round:
	$(MAKE) deploy WITH_DECIDER=true
	$(MAKE) agent
	$(MAKE) deploy WITH_DECIDER=true AGENT_KEY=$$(cat /tmp/ch10-agent.key)
	$(MAKE) image
	@# The repository URI is read here by the shell, not by make. Make expands
	@# a whole recipe before running its first line, so DECIDER_REPO would be
	@# looked up before the deploy above had created the stack to look it up
	@# in, and the image would deploy as ":<tag>" with no registry in front.
	$(MAKE) deploy WITH_DECIDER=true AGENT_KEY=$$(cat /tmp/ch10-agent.key) \
		DECIDER_IMAGE=$$(aws cloudformation describe-stacks --region $(REGION) \
		--stack-name $(STACK) --output text \
		--query "Stacks[0].Outputs[?OutputKey=='DeciderRepoUri'].OutputValue"):$$(cat /tmp/ch10-image.tag)
	$(MAKE) decide
	$(MAKE) route
	$(MAKE) eval

# image: the model's inference container, built in the account by CodeBuild.
# x86_64 with CUDA torch and the weights baked in, so SageMaker can answer
# inside its 60 second limit on the first request.
image:
	rm -rf /tmp/ch10-build && mkdir -p /tmp/ch10-build
	cp ../model/Dockerfile ../model/serve.py ../model/entrypoint.sh /tmp/ch10-build/
	cp buildspec.yml /tmp/ch10-build/
	rm -f /tmp/ch10-src.zip && cd /tmp/ch10-build && zip -qr /tmp/ch10-src.zip . -x '*__pycache__*'
	@BUCKET=$$(aws cloudformation describe-stacks --region $(REGION) --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='DeciderSourceBucket'].OutputValue" --output text); \
		aws s3 cp /tmp/ch10-src.zip s3://$$BUCKET/src.zip --region $(REGION); \
		TAG=img-$$(shasum -a 256 /tmp/ch10-src.zip | cut -c1-12); \
		echo $$TAG > /tmp/ch10-image.tag; \
		ID=$$(aws codebuild start-build --region $(REGION) --project-name ch10-decider-build \
			--environment-variables-override name=TAG,value=$$TAG,type=PLAINTEXT \
			--query 'build.id' --output text); \
		echo "build $$ID"; \
		while STATUS=$$(aws codebuild batch-get-builds --region $(REGION) --ids $$ID \
			--query 'builds[0].buildStatus' --output text); \
			[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 20; echo "  $$STATUS"; done; \
			echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]; \
			echo "pushed $$(cat /tmp/ch10-image.tag)"

# agent: the routing agent as a zip, for direct code deployment. arm64 wheels,
# because AgentCore Runtime is arm64 only. categories.py travels with it, so the
# agent and the LLM baseline are briefed from one text.
agent:
	rm -rf /tmp/ch10-agent && mkdir -p /tmp/ch10-agent
	uv pip install --python-platform aarch64-manylinux2014 --python-version 3.12 \
		--target=/tmp/ch10-agent --only-binary=:all: -r ../agent/pyproject.toml
	cp ../agent/main.py ../src/categories.py /tmp/ch10-agent/
	rm -f /tmp/ch10-agent.zip && cd /tmp/ch10-agent && zip -qr /tmp/ch10-agent.zip . -x '*__pycache__*'
	@# The key carries the zip's own hash. CloudFormation updates a runtime when
	@# its parameters change, and replacing the object behind an unchanged key is
	@# not a change: the old code keeps serving and the deploy looks successful.
	@BUCKET=$$(aws cloudformation describe-stacks --region $(REGION) --stack-name $(STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='DeciderSourceBucket'].OutputValue" --output text); \
			KEY=agent-$$(shasum -a 256 /tmp/ch10-agent.zip | cut -c1-12).zip; \
			aws s3 cp /tmp/ch10-agent.zip s3://$$BUCKET/$$KEY --region $(REGION); \
			echo $$KEY > /tmp/ch10-agent.key; \
			echo "uploaded $$KEY ($$(du -h /tmp/ch10-agent.zip | cut -f1))"

# decide: route the dataset through the agent, which asks the endpoint.
# botocore[crt]: the profile chain behind `aws login` needs it, and a fresh
# uv environment does not pull it in with boto3 alone.
decide:
	cd .. && PYTHONPATH=src DECIDER_BACKEND=agentcore AWS_DEFAULT_REGION=$(REGION) \
		AGENT_RUNTIME_ARN=$(DECIDER_ARN) \
		uv run --with boto3 --with "botocore[crt]" src/decide.py --dataset $(DATASET) --variant b

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
