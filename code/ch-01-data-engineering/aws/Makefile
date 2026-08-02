.PHONY: iam airflow-deploy airflow-password airflow-delete lake-objects lake-parquet lake-table glue-run feature-group clean-feature-group check-account clean-lake

S3TABLES_ARN ?= arn:aws:s3tables:us-east-1:$(ACCOUNT):bucket/ch01-lake

# Cloud verification: the SAME scripts as the local targets, run against
# real AWS through the default credential chain. Bucket names get the
# account id suffix because the S3 namespace is global.
ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text)
LAKE_BUCKET ?= credit-lake-$(ACCOUNT)
RAW_BUCKET ?= bureau-raw-$(ACCOUNT)
RUN_STAMP := $(shell date +%Y-%m-%d)

# Airflow on EC2 (Postgres metadata + LocalExecutor, nightly pg_dump to S3).
# AllowedCidr is set to your current public IP so only you reach the UI on :8080;
# stack name, region, and capabilities come from airflow-ec2/samconfig.toml.
# WAREHOUSE=local (default, free tier: redshift-local on the instance) or
# WAREHOUSE=serverless (SAM also provisions Redshift Serverless in this stack —
# pass SUBNETS=subnet-a,subnet-b,subnet-c across >= 3 AZs for the workgroup).
WAREHOUSE ?= local
SUBNETS ?=
airflow-deploy:
	cd airflow-ec2 && sam deploy --no-fail-on-empty-changeset \
		--parameter-overrides AllowedCidr=$$(curl -4 -s ifconfig.me)/32 \
		WarehouseMode=$(WAREHOUSE) $(if $(SUBNETS),WorkgroupSubnetIds=$(SUBNETS),)

# Airflow 3 standalone generates the admin password on the instance; fetch it
# over SSM (no SSH, no extra open port). Log in to the UI as user "admin".
airflow-password:
	@IID=$$(aws ec2 describe-instances --filters Name=tag:Name,Values=ch01-airflow \
	  Name=instance-state-name,Values=running \
	  --query 'Reservations[0].Instances[0].InstanceId' --output text); \
	  CID=$$(aws ssm send-command --instance-ids $$IID --document-name AWS-RunShellScript \
	  --parameters commands='docker exec af-airflow-1 cat /opt/airflow/simple_auth_manager_passwords.json.generated' \
	  --query Command.CommandId --output text); \
	  aws ssm wait command-executed --command-id $$CID --instance-id $$IID 2>/dev/null || true; \
	  aws ssm get-command-invocation --command-id $$CID --instance-id $$IID --query StandardOutputContent --output text

# S3 blocks CloudFormation from deleting a non-empty bucket, so empty the
# backup bucket (read from the stack outputs) before sam delete tears it down.
airflow-delete:
	@BUCKET=$$(aws cloudformation describe-stacks --stack-name ch01-airflow \
	  --query "Stacks[0].Outputs[?OutputKey=='BackupBucket'].OutputValue" \
	  --output text 2>/dev/null); \
	if [ -n "$$BUCKET" ] && [ "$$BUCKET" != "None" ]; then \
		echo "emptying s3://$$BUCKET"; aws s3 rm "s3://$$BUCKET" --recursive; \
	fi
	cd airflow-ec2 && sam delete --no-prompts

check-account:
	@echo "account: $(ACCOUNT)  lake bucket: $(LAKE_BUCKET)"

lake-objects:
	cd .. && env -u AWS_ENDPOINT_URL -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		LAKE_BUCKET=$(LAKE_BUCKET) uv run src/lake-basics/s3_objects.py \
		| tee /tmp/ch01-lake-objects-$(RUN_STAMP).txt

lake-parquet:
	cd .. && env -u AWS_ENDPOINT_URL -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		LAKE_BUCKET=$(LAKE_BUCKET) uv run src/lake-basics/parquet_lake.py \
		| tee /tmp/ch01-lake-parquet-$(RUN_STAMP).txt

lake-table:
	cd .. && env -u AWS_ENDPOINT_URL -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		S3TABLES_ARN=$(S3TABLES_ARN) uv run src/lake-basics/iceberg_applications.py \
		| tee /tmp/ch01-lake-table-$(RUN_STAMP).txt

glue-run:
	# the bureau job as a real Glue job (see aws/README.md for the one-time setup)
	aws glue start-job-run --job-name bureau-elt-ch01 --arguments \
		'{"--date":"2026-07-17","--destination":"filesystem","--raw-bucket":"$(RAW_BUCKET)","--bucket-url":"s3://$(RAW_BUCKET)/bronze","--additional-python-modules":"dlt,s3fs"}' \
		--query 'JobRunId' --output text | tee /tmp/ch01-glue-run-$(RUN_STAMP).txt

feature-group:
	cd .. && env -u AWS_ENDPOINT_URL -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		uv run src/feature-store/extra/feature_group_demo.py --applicant 10046 \
		| tee /tmp/ch01-feature-group-$(RUN_STAMP).txt

clean-feature-group:
	env -u AWS_ENDPOINT_URL aws sagemaker delete-feature-group --feature-group-name applicant-credit-profile-fg

clean-lake:
	env -u AWS_ENDPOINT_URL aws s3 rm s3://$(LAKE_BUCKET) --recursive
	env -u AWS_ENDPOINT_URL aws s3 rb s3://$(LAKE_BUCKET)

# -------------------------------------------------------------------------------
# iam: provision THIS chapter's deploy identity, when you start the chapter
# Creates ch01-user from iam/deploy.json (least-privilege for ch-01 only) and trusts you to
# assume it — per chapter, not an account-wide up-front bootstrap. Re-run after
# editing the policy. Then add a [profile ch01] (role_arn + source_profile) and
# run cloud targets with AWS_PROFILE=ch01.
# -------------------------------------------------------------------------------
PRINCIPAL ?= $(shell aws sts get-caller-identity --query Arn --output text)
TRUST = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"$(PRINCIPAL)"},"Action":"sts:AssumeRole"}]}
iam:
	@aws iam create-role --role-name ch01-user --description "Deploy/operate ch-01" \
		--assume-role-policy-document '$(TRUST)' 2>/dev/null \
		|| aws iam update-assume-role-policy --role-name ch01-user --policy-document '$(TRUST)'
	aws iam put-role-policy --role-name ch01-user --policy-name deploy --policy-document file://iam/deploy.json
	@echo "ch01-user ready"