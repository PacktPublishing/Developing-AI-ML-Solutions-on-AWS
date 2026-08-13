.PHONY: iam check-account sync build deploy round teardown

# the AWS round in one command (run make iam once first, as the account admin): zip and sync
# the app, then deploy the Aurora Serverless express stack and the assistant Lambda behind its
# Function URL. make teardown when done.
round: deploy

STACK ?= underwriting-assistant-ch03
REGION ?= us-east-1

# `make deploy` builds and deploys the assistant SAM stack from this directory.
ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text)

check-account:
	@echo "account: $(ACCOUNT)"

# corpus_data.py has one source of truth under ../src; the build copies it into
# src/ (src/corpus_data.py is gitignored) so there is no drift.
sync:
	cp ../src/corpus_data.py src/corpus_data.py

build: sync
	sam build --beta-features

deploy: build
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset \
		--capabilities CAPABILITY_IAM --region $(REGION)

teardown:
	sam delete --stack-name $(STACK) --region $(REGION) --no-prompts

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch03-user --description "Deploy/operate ch-03" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch03-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch03-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch03-user ready"