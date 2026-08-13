.PHONY: iam lint seed-warehouse deploy image endpoint-local endpoint produce blocks query round teardown

# The cloud round in order: deploy, seed-warehouse, image, endpoint, produce,
# blocks, query, teardown. EnableWarehouse=true (the default) builds the warehouse.
REGION ?= us-east-1
STACK := streaming-fraud-ch06

# stack outputs, resolved on demand
output = $(shell aws cloudformation describe-stacks --region $(REGION) \
	--stack-name $(STACK) \
	--query "Stacks[0].Outputs[?OutputKey=='$(1)'].OutputValue" --output text)

# the whole AWS round in one command (run make iam once first, as the account admin): the
# stack, the scoring image, the serverless endpoint, a replay of transactions, then the
# blocked count and the warehouse tally. The consumer (Lambda) and Firehose are asynchronous,
# so the round waits between the replay and the reads. make teardown when done.
round:
	$(MAKE) deploy
	$(MAKE) seed-warehouse
	$(MAKE) image
	$(MAKE) endpoint
	$(MAKE) produce
	@echo "waiting for the consumer to score + block and Firehose to deliver..."; sleep 150
	$(MAKE) blocks
	@sleep 90
	$(MAKE) query

# template checks: cfn-lint for correctness, rain for canonical formatting
lint:
	uvx cfn-lint template.yaml
	rain fmt --verify template.yaml

deploy:
	sam build
	sam deploy --no-confirm-changeset --no-fail-on-empty-changeset

# create the destination table via the Redshift Data API, no VPC path needed,
# authenticated with the admin secret the stack generated
seed-warehouse:
	$(eval WG := $(call output,WorkgroupName))
	$(eval SECRET := $(call output,WarehouseSecretArn))
	@ID=$$(aws redshift-data execute-statement --region $(REGION) \
		--workgroup-name $(WG) --database fraud --secret-arn $(SECRET) \
		--sql "CREATE TABLE IF NOT EXISTS fraud_decisions (transaction_id VARCHAR(32), user_id VARCHAR(16), merchant_id VARCHAR(16), amount_usd NUMERIC(12,2), score DOUBLE PRECISION, decision VARCHAR(8), event_time TIMESTAMP, latency_ms DOUBLE PRECISION)" \
		--query Id --output text); \
		while S=$$(aws redshift-data describe-statement --region $(REGION) --id $$ID --query Status --output text); \
			[ "$$S" = "SUBMITTED" ] || [ "$$S" = "PICKED" ] || [ "$$S" = "STARTED" ]; do sleep 1; done; \
			echo "fraud_decisions $$S"; [ "$$S" = "FINISHED" ]

# build the serving image via the stack's CodeBuild project (multi-arch push to ECR).
image:
	$(eval BUCKET := $(call output,StagingBucketName))
	cd .. && rm -f /tmp/ch06-scoring-src.zip && \
		zip -q -j /tmp/ch06-scoring-src.zip src/scoring/Dockerfile src/scoring/serve.py \
		src/scoring/entrypoint.sh aws/image/buildspec.yml
	aws s3 cp /tmp/ch06-scoring-src.zip s3://$(BUCKET)/ch06/source/scoring.zip
	@ID=$$(aws codebuild start-build --region $(REGION) --project-name ch06-scoring-build \
		--query 'build.id' --output text); \
	echo "build $$ID"; \
	while STATUS=$$(aws codebuild batch-get-builds --region $(REGION) --ids $$ID \
		--query 'builds[0].buildStatus' --output text); \
		[ "$$STATUS" = "IN_PROGRESS" ]; do sleep 15; echo "  $$STATUS"; done; \
		echo "$$STATUS"; [ "$$STATUS" = "SUCCEEDED" ]

# rehearse the deploy in SageMaker local mode: the same amd64 endpoint image, container runs
# here (on an Apple-silicon laptop it runs under emulation, so the first /ping is slow).
endpoint-local:
	$(eval REPO := $(call output,ScoringRepoUri))
	$(eval BUCKET := $(call output,StagingBucketName))
	$(eval ROLE := $(call output,EndpointRoleArn))
	cd .. && SAGEMAKER_ROLE_ARN=$(ROLE) IMAGE_URI=$(REPO):latest \
		ARTIFACT_BUCKET=$(BUCKET) AWS_DEFAULT_REGION=$(REGION) \
		uv run aws/deploy.py

# the real deploy: model.tar.gz to S3, then model + endpoint config + endpoint.
endpoint:
	$(eval REPO := $(call output,ScoringRepoUri))
	$(eval BUCKET := $(call output,StagingBucketName))
	$(eval ROLE := $(call output,EndpointRoleArn))
	cd .. && SAGEMAKER_SERVERLESS=1 SAGEMAKER_ROLE_ARN=$(ROLE) IMAGE_URI=$(REPO):latest \
		ARTIFACT_BUCKET=$(BUCKET) AWS_DEFAULT_REGION=$(REGION) \
		uv run aws/deploy.py

# replay transactions at the real stream: same producer, endpoint variables
# cleared so boto3 talks to AWS
produce:
	cd .. && KINESIS_ENDPOINT= uv run src/streaming/producer.py --limit 500

# how many transactions the consumer blocked
blocks:
	aws dynamodb scan --region $(REGION) --table-name blocked_transactions \
		--select COUNT --query Count

# what the analyst sees in the warehouse: blocks and passes, via the Data API
query:
	$(eval WG := $(call output,WorkgroupName))
	$(eval SECRET := $(call output,WarehouseSecretArn))
	@ID=$$(aws redshift-data execute-statement --region $(REGION) \
		--workgroup-name $(WG) --database fraud --secret-arn $(SECRET) \
		--sql "SELECT decision, COUNT(*), ROUND(AVG(score)::numeric, 4) FROM fraud_decisions GROUP BY decision" \
		--query Id --output text); \
		while S=$$(aws redshift-data describe-statement --region $(REGION) --id $$ID --query Status --output text); \
			[ "$$S" = "SUBMITTED" ] || [ "$$S" = "PICKED" ] || [ "$$S" = "STARTED" ]; do sleep 1; done; \
			aws redshift-data get-statement-result --region $(REGION) --id $$ID --output json

# delete in reverse: endpoint first (it is not in the stack), then empty the
# staging bucket, then the stack, warehouse and all
teardown:
	-aws sagemaker delete-endpoint --region $(REGION) --endpoint-name fraud-scoring
	-aws sagemaker delete-endpoint-config --region $(REGION) --endpoint-config-name fraud-scoring
	-aws sagemaker delete-model --region $(REGION) --model-name fraud-scoring
	-aws s3 rm --recursive s3://$(call output,StagingBucketName)
	sam delete --no-prompts --region $(REGION)

# -------------------------------------------------------------------------------
# iam: provision this chapter's deploy identity (ch06-user from iam/deploy.json)
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch06-user --description "Deploy/operate ch-06" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch06-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch06-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch06-user ready"