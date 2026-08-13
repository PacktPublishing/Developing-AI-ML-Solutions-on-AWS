.PHONY: iam secret source deploy image seed provision scale-up scale-down round teardown

# the whole AWS round in one command (run make iam once first, as the account admin): the
# stack, the container image, the warehouse seed, the security model (read-only accounts +
# engine-side masking), and the service scaled up for a session. Open the task's public IP
# on :7681 to ask. make scale-down (park at 0) or teardown when done.
round:
	$(MAKE) deploy
	$(MAKE) image
	$(MAKE) seed
	$(MAKE) provision
	$(MAKE) scale-up

REGION := us-east-1
STACK := ch08-selfservice-analytics
ACCOUNT = $(shell aws sts get-caller-identity --query Account --output text)
REPO := ch08-assistant
IMAGE = $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com/$(REPO):latest
WORKGROUP := ch08-analytics
DB := analytics
SECRET := ch08-admin-password
SOURCE_BUCKET = ch08-source-$(ACCOUNT)

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) --query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)

# the template takes an existing network: the default VPC, its public subnets, and your IP
VPC = $(shell aws ec2 describe-vpcs --region $(REGION) --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNETS = $(shell aws ec2 describe-subnets --region $(REGION) --filters Name=vpc-id,Values=$(VPC) Name=map-public-ip-on-launch,Values=true --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
MYIP = $(shell curl -s https://checkip.amazonaws.com)/32

# -------------------------------------------------------------------------------
# deploy: source zip -> stack -> (image | seed) -> provision -> scale up
# -------------------------------------------------------------------------------
# secret: the Redshift admin password the template resolves at deploy time
secret:
	@aws secretsmanager describe-secret --secret-id $(SECRET) --region $(REGION) >/dev/null 2>&1 || \
		aws secretsmanager create-secret --name $(SECRET) --region $(REGION) --secret-string \
		"$$(aws secretsmanager get-random-password --region $(REGION) --exclude-punctuation --password-length 32 --query RandomPassword --output text)" >/dev/null
	@echo "secret $(SECRET) ready"

# source: the chapter zip CodeBuild builds the image from
source:
	@aws s3api head-bucket --bucket $(SOURCE_BUCKET) 2>/dev/null || aws s3 mb s3://$(SOURCE_BUCKET) --region $(REGION)
	cd .. && rm -f /tmp/ch08-source.zip && \
		zip -qr /tmp/ch08-source.zip Dockerfile CLAUDE.md pyproject.toml launch-claude \
		docker-entrypoint.sh skills config aws/image/buildspec.yml -x '*/__pycache__/*'
	aws s3 cp /tmp/ch08-source.zip s3://$(SOURCE_BUCKET)/ch08/source.zip

# deploy: the whole stack -- Redshift Serverless, ECR, CodeBuild, ECS at desired 0
deploy: secret source
	sam deploy --stack-name $(STACK) --region $(REGION) --resolve-s3 \
		--capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
		--parameter-overrides VpcId=$(VPC) SubnetIds=$(SUBNETS) AllowedCidr=$(MYIP) \
		ImageUri=$(IMAGE) SourceBucket=$(SOURCE_BUCKET)

# image and seed are independent once the stack is up: build the container in
# the cloud while the warehouse loads. Overlap them with make -j2 image seed.
image:
	@ID=$$(aws codebuild start-build --region $(REGION) --project-name ch08-assistant \
		--query 'build.id' --output text); \
	echo "build $$ID"; \
	while STATUS=$$(aws codebuild batch-get-builds --region $(REGION) --ids $$ID \
		--query 'builds[0].buildStatus' --output text); \
		[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
		echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

seed:
	REDSHIFT_WORKGROUP=$(WORKGROUP) REDSHIFT_DB_NAME=$(DB) REDSHIFT_REGION=$(REGION) \
		uv run --no-project seed.py

# provision: the security model -- read-only IAM accounts + engine-side masking.
# TASK_ROLE lets provision grant and mask the assistant's own Data API identity
# (IAMR:<task role>), not just the named bi_analyst account.
provision:
	REDSHIFT_WORKGROUP=$(WORKGROUP) REDSHIFT_DB_NAME=$(DB) REDSHIFT_REGION=$(REGION) \
		TASK_ROLE=$(call output,TaskRoleName) bash ../config/provision.sh

# scale the Fargate service to 1 for a session, back to 0 after (nothing bills parked)
scale-up:
	aws ecs update-service --region $(REGION) --cluster ch08-analytics \
		--service $(call output,ServiceName) --desired-count 1 >/dev/null
	aws ecs wait services-stable --region $(REGION) --cluster ch08-analytics \
		--services $(call output,ServiceName)
	@echo "service up; open the task's public IP on :7681 (allowed to $(MYIP))"

scale-down:
	aws ecs update-service --region $(REGION) --cluster ch08-analytics \
		--service $(call output,ServiceName) --desired-count 0 >/dev/null

teardown:
	-aws ecs update-service --region $(REGION) --cluster ch08-analytics \
		--service $(call output,ServiceName) --desired-count 0
	-aws s3 rm s3://$(SOURCE_BUCKET) --recursive
	sam delete --stack-name $(STACK) --region $(REGION) --no-prompts
	-aws s3 rb s3://$(SOURCE_BUCKET)
	-aws secretsmanager delete-secret --secret-id $(SECRET) --region $(REGION) --force-delete-without-recovery
	@echo "$(STACK) torn down"

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# -------------------------------------------------------------------------------
# Creates ch08-user from iam/deploy.json and trusts you to assume it; re-run after
# editing the policy, then add a [profile ch08] and run cloud targets AWS_PROFILE=ch08.
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch08-user --description "Deploy/operate ch-08" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch08-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch08-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch08-user ready"