.PHONY: iam deploy classify route eval round teardown

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

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) --query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)
BUCKET = $(call output,BatchBucket)
ROLE = $(call output,BatchServiceRoleArn)

# The cloud round talks to the real services: no BEDROCK_LOCAL, no S3_ENDPOINT.
CLOUDENV = PYTHONPATH=src BEDROCK_REGION=$(REGION) BATCH_MODEL=$(BATCH_MODEL) \
	BEDROCK_BATCH_ROLE=$(ROLE)

# -------------------------------------------------------------------------------
# deploy -> classify (real batch job) -> route (real SNS) -> eval
# -------------------------------------------------------------------------------
deploy:
	aws cloudformation deploy --region $(REGION) --stack-name $(STACK) \
		--template-file template.yaml --no-fail-on-empty-changeset \
		--capabilities CAPABILITY_NAMED_IAM

# classify: submit a real CreateModelInvocationJob and wait for it (tens of minutes)
classify:
	cd .. && $(CLOUDENV) uv run src/classify.py --dataset $(DATASET) --bucket $(BUCKET)

# route: publish each classified conversation to its team's real SNS topic
route:
	cd .. && PYTHONPATH=src BEDROCK_REGION=$(REGION) uv run src/route.py --dataset $(DATASET)

eval:
	cd .. && PYTHONPATH=src uv run src/evaluate.py --dataset $(DATASET)

teardown:
	-aws s3 rm s3://$(BUCKET) --recursive --region $(REGION)
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