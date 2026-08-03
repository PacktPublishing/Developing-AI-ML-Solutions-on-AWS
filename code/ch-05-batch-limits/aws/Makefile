.PHONY: iam lint deploy warehouse warehouse-down seed image pipeline run-now teardown

REGION ?= us-east-1
STACK := batch-limits-ch05

output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) \
	--query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)

lint:
	uvx cfn-lint template.yaml
	rain fmt --verify template.yaml

deploy:
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset

# add the warehouse + its self-contained VPC for the run. It bills per second, so
# `make warehouse-down` (DeployWarehouse=false) or `make teardown` when finished.
# Nothing to look up: the template owns the VPC, subnets, and endpoints.
warehouse:
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset \
		--parameter-overrides DeployWarehouse=true

warehouse-down:
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset \
		--parameter-overrides DeployWarehouse=false

# seed the private warehouse over the Data API (upload the CSV, then COPY FROM S3)
seed:
	cd .. && WORKGROUP=$(call output,WorkgroupName) \
		BUCKET=$(call output,BatchBucketName) \
		COPY_ROLE_ARN=$(call output,CopyRoleArn) \
		AWS_DEFAULT_REGION=$(REGION) uv run aws/jobs/seed_redshift.py

# build the step image in the cloud and push it to the stack's ECR repo
image:
	$(eval BUCKET := $(call output,BatchBucketName))
	cd .. && rm -f /tmp/ch05-steps-src.zip && \
		zip -q -j /tmp/ch05-steps-src.zip src/pipeline/step_image/Dockerfile \
		src/pipeline/step_image/train aws/image/buildspec.yml
	aws s3 cp /tmp/ch05-steps-src.zip s3://$(BUCKET)/ch05/source/steps.zip
	@ID=$$(aws codebuild start-build --region $(REGION) --project-name ch05-step-build \
		--query 'build.id' --output text); \
	echo "build $$ID"; \
	while STATUS=$$(aws codebuild batch-get-builds --region $(REGION) --ids $$ID \
		--query 'builds[0].buildStatus' --output text); \
		[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
		echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

# register the pipeline on AWS: same definition, PIPELINE_MODE=aws. The DSN
# (endpoint + generated password) and the VpcConfig (subnets + job SG) are all
# read from the stack outputs — nothing to pass by hand.
pipeline:
	$(eval REPO := $(call output,StepRepoUri))
	$(eval BUCKET := $(call output,BatchBucketName))
	$(eval ROLE := $(call output,PipelineRoleArn))
	$(eval ENDPOINT := $(call output,WarehouseEndpoint))
	$(eval SUBNETS := $(call output,JobSubnetIds))
	$(eval JOBSG := $(call output,JobSecurityGroupId))
	$(eval PW := $(shell aws secretsmanager get-secret-value --region $(REGION) \
		--secret-id $(call output,WarehouseSecretArn) --query SecretString --output text \
		| python3 -c "import sys,json;print(json.load(sys.stdin)['password'])"))
	cd .. && PIPELINE_MODE=aws AWS_DEFAULT_REGION=$(REGION) \
		STEP_IMAGE=$(REPO):latest BATCH_BUCKET=$(BUCKET) SAGEMAKER_ROLE_ARN=$(ROLE) \
		WAREHOUSE_DSN="postgresql://analyst:$(PW)@$(ENDPOINT):5439/portfolio?sslmode=require" \
		NETWORK_SUBNETS=$(SUBNETS) NETWORK_SG=$(JOBSG) \
		uv run src/pipeline/pipeline.py

# start one execution without waiting for the schedule
run-now:
	aws sagemaker start-pipeline-execution --region $(REGION) \
		--pipeline-name batch-credit-limits --query 'PipelineExecutionArn'

teardown:
	-aws sagemaker delete-pipeline --region $(REGION) --pipeline-name batch-credit-limits
	-aws s3 rm --recursive s3://$(call output,BatchBucketName)
	sam delete --no-prompts --region $(REGION)

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# Creates ch05-user from iam/deploy.json (least-privilege for ch-05 only) and trusts you to
# assume it — per chapter, not an account-wide up-front bootstrap. Re-run after
# editing the policy. Then add a [profile ch05] (role_arn + source_profile) and
# run cloud targets with AWS_PROFILE=ch05.
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch05-user --description "Deploy/operate ch-05" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch05-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch05-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch05-user ready"