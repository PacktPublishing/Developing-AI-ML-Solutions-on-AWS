.PHONY: airflow-deploy airflow-delete lake-objects lake-parquet lake-table glue-run feature-group clean-feature-group check-account clean-lake

S3TABLES_ARN ?= arn:aws:s3tables:us-east-1:$(ACCOUNT):bucket/book-lake-ch01

# Cloud verification: the SAME scripts as the local targets, run against
# real AWS through the default credential chain. Bucket names get the
# account id suffix because the S3 namespace is global.
ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text)
LAKE_BUCKET ?= credit-lake-$(ACCOUNT)
RAW_BUCKET ?= bureau-raw-$(ACCOUNT)
RUN_STAMP := $(shell date +%Y-%m-%d)

airflow-deploy:
	cd airflow-ec2 && sam deploy --parameter-overrides AllowedCidr=$$(curl -4 -s ifconfig.me)/32 --no-fail-on-empty-changeset

airflow-delete:
	cd airflow-ec2 && sam delete --no-prompts

check-account:
	@echo "account: $(ACCOUNT)  lake bucket: $(LAKE_BUCKET)"

lake-objects:
	cd .. && env -u AWS_ENDPOINT_URL -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		LAKE_BUCKET=$(LAKE_BUCKET) uv run lake-basics/s3_objects.py \
		| tee /tmp/ch01-lake-objects-$(RUN_STAMP).txt

lake-parquet:
	cd .. && env -u AWS_ENDPOINT_URL -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		LAKE_BUCKET=$(LAKE_BUCKET) uv run lake-basics/parquet_lake.py \
		| tee /tmp/ch01-lake-parquet-$(RUN_STAMP).txt

lake-table:
	cd .. && env -u AWS_ENDPOINT_URL -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		S3TABLES_ARN=$(S3TABLES_ARN) uv run lake-basics/iceberg_applications.py \
		| tee /tmp/ch01-lake-table-$(RUN_STAMP).txt

glue-run:
	# the bureau job as a real Glue job (see aws/README.md for the one-time setup)
	aws glue start-job-run --job-name bureau-elt-ch01 --arguments \
		'{"--date":"2026-07-17","--destination":"filesystem","--raw-bucket":"$(RAW_BUCKET)","--bucket-url":"s3://$(RAW_BUCKET)/bronze","--additional-python-modules":"dlt,s3fs"}' \
		--query 'JobRunId' --output text | tee /tmp/ch01-glue-run-$(RUN_STAMP).txt

feature-group:
	cd .. && env -u AWS_ENDPOINT_URL -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		uv run feature-store/extra/feature_group_demo.py --applicant 10046 \
		| tee /tmp/ch01-feature-group-$(RUN_STAMP).txt

clean-feature-group:
	env -u AWS_ENDPOINT_URL aws sagemaker delete-feature-group --feature-group-name applicant-credit-profile-fg

clean-lake:
	env -u AWS_ENDPOINT_URL aws s3 rm s3://$(LAKE_BUCKET) --recursive
	env -u AWS_ENDPOINT_URL aws s3 rb s3://$(LAKE_BUCKET)