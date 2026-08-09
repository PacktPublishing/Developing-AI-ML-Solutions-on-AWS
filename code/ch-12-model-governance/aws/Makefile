.PHONY: iam deploy flip teardown

# The governance round on AWS: AppConfig drives the rollout, the AppConfig agent sidecar
# on a Fargate gateway reads it, and SageMaker holds the model cards, registry, and
# lineage. Account values are derived; override on the command line.
REGION ?= us-east-1
ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text)
STACK := ch12-governance
PCT ?= 20

# -------------------------------------------------------------------------------
# iam: provision this chapter's deploy identity (run once, as the account admin)
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch12-user --description "Deploy/operate ch-12" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch12-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch12-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch12-user ready: arn:aws:iam::$(ACCOUNT):role/ch12-user"

# -------------------------------------------------------------------------------
# deploy: AppConfig (application, environment, feature-flag config) + the ECR repo
# -------------------------------------------------------------------------------
deploy:
	aws cloudformation deploy --template-file template.yaml --stack-name $(STACK) \
		--capabilities CAPABILITY_NAMED_IAM \
		--parameter-overrides RolloutPct=$(PCT)
	@echo "stack $(STACK) ready (rollout $(PCT)%)"

# widen the rollout on AWS: a new configuration version + deployment, service untouched
flip:
	aws cloudformation deploy --template-file template.yaml --stack-name $(STACK) \
		--capabilities CAPABILITY_NAMED_IAM \
		--parameter-overrides RolloutPct=$(PCT)
	@echo "rollout -> $(PCT)%"

teardown:
	aws cloudformation delete-stack --stack-name $(STACK)
	aws cloudformation wait stack-delete-complete --stack-name $(STACK)
	@echo "$(STACK) torn down"