.PHONY: iam image deploy flip teardown

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
# image: build the two model images in the cloud (CodeBuild -> ECR), natively on amd64
# -------------------------------------------------------------------------------
# build.yaml creates the source bucket, the two ECR repos, and the CodeBuild project;
# then zip chapter 2's scorecard + challenger contexts with buildspec.yml at the root,
# upload, run the build, and poll to SUCCEEDED.
BUILD_STACK := ch12-image-build
image:
	aws cloudformation deploy --template-file build.yaml --stack-name $(BUILD_STACK) --capabilities CAPABILITY_IAM
	rm -f /tmp/ch12-src.zip
	cd ../../ch-02-mlops/src && zip -qr /tmp/ch12-src.zip scorecard challenger -x '*__pycache__*'
	zip -qj /tmp/ch12-src.zip buildspec.yml
	@BUCKET=$$(aws cloudformation describe-stacks --stack-name $(BUILD_STACK) \
			--query "Stacks[0].Outputs[?OutputKey=='SourceBucket'].OutputValue" --output text); \
		aws s3 cp /tmp/ch12-src.zip s3://$$BUCKET/src.zip; \
		ID=$$(aws codebuild start-build --project-name ch12-models-build --query 'build.id' --output text); \
		echo "build $$ID"; \
		while STATUS=$$(aws codebuild batch-get-builds --ids $$ID --query 'builds[0].buildStatus' --output text); \
			[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
			echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

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