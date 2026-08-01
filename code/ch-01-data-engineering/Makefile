.PHONY: test up seed seed-drift run run-glue query dbt-run dbt-test gold features features-query features-lookup feature-api feature-group lake-objects lake-parquet lake-table lake-table-query diagrams down lint

# dbt connects with verified TLS: oblako-ca.pem is the local warehouse cert,
# trusted via SSL_CERT_FILE, same sslmode as production
DBT := DBT_PROFILES_DIR=src/bureau-elt/dbt SSL_CERT_FILE=$(PWD)/oblako-bundle.pem WAREHOUSE_SSLMODE=verify-full uvx --from dbt-core --with dbt-redshift dbt

DATE ?= $(shell date -v-1d +%Y-%m-%d 2>/dev/null || date -d yesterday +%Y-%m-%d)

export AWS_ENDPOINT_URL ?= http://localhost:9000
export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1
export DLT_DESTINATION ?= redshift
# dlt parses this as a Postgres DSN; the redshift destination is chosen by
# DLT_DESTINATION, not by the URL scheme
export DESTINATION__REDSHIFT__CREDENTIALS ?= postgresql://loader:loader@localhost:5439/bureau?sslmode=require

COMPOSE := docker compose -f local/lakehouse.yml

up:
	$(COMPOSE) up -d --wait

seed:
	uv run src/bureau-elt/generate_bureau_files.py --date $(DATE)

seed-drift:
	uv run src/bureau-elt/generate_bureau_files.py --date $(DATE) --drift

run:
	uv run src/bureau-elt/glue_bureau_job.py --date $(DATE)

run-glue:
	# the same job inside the official Glue runtime image; pip install at
	# start mirrors what --additional-python-modules does on AWS
	docker run --rm -v "$(PWD)":/work \
		-e AWS_ENDPOINT_URL=http://host.docker.internal:9000 \
		-e AWS_ACCESS_KEY_ID=local -e AWS_SECRET_ACCESS_KEY=localsecret \
		-e AWS_DEFAULT_REGION=us-east-1 -e DLT_DESTINATION=redshift \
		-e DESTINATION__REDSHIFT__CREDENTIALS=postgresql://loader:loader@host.docker.internal:5439/bureau \
		--entrypoint bash public.ecr.aws/glue/aws-glue-libs:5 \
		-c "pip install -q 'dlt[redshift]' && python3 /work/src/bureau-elt/glue_bureau_job.py --date $(DATE)"

query:
	$(COMPOSE) exec warehouse psql -h 127.0.0.1 -p 5433 -U loader -d bureau -c \
		"select count(*) as reports from bureau_raw.reports;" -c \
		"select count(*) as tradelines from bureau_raw.reports__tradelines;" -c \
		"select column_name from information_schema.columns where table_name = 'reports' order by ordinal_position;"

oblako-ca.pem:
	openssl s_client -starttls postgres -connect localhost:5439 </dev/null 2>/dev/null | openssl x509 -outform pem > oblako-ca.pem

# system CAs plus the warehouse cert: redshift_connector verifies the local
# warehouse while uv itself can still verify PyPI
oblako-bundle.pem: oblako-ca.pem
	uv run --with certifi python -c "import certifi, shutil; shutil.copy(certifi.where(), 'oblako-bundle.pem')"
	cat oblako-ca.pem >> oblako-bundle.pem

dbt-run: oblako-bundle.pem
	$(DBT) run --project-dir src/bureau-elt/dbt

dbt-test: oblako-bundle.pem
	$(DBT) test --project-dir src/bureau-elt/dbt

gold:
	$(COMPOSE) exec warehouse psql -h 127.0.0.1 -p 5433 -U loader -d bureau -c \
		"select * from bureau_marts.applicant_credit_profile order by score desc limit 10;"

lake-objects:
	uv run src/lake-basics/s3_objects.py

lake-parquet:
	uv run src/lake-basics/parquet_lake.py

lake-table:
	uv run src/lake-basics/iceberg_applications.py

lake-table-query:
	$(COMPOSE) exec trino trino --execute \
		"select vintage, decision, count(*) as n from iceberg.lake.applications group by 1, 2 order by 1, 2"

features:
	uv run src/feature-store/ingest_features.py

features-query:
	$(COMPOSE) exec trino trino --execute \
		"select applicant_id, score, tradeline_count, total_balance from iceberg.features.applicant_credit_profile order by score desc limit 5"

# Regenerate the architecture figures from diagrams/yaml/ with a pinned awsdac.
# AWS icons are fetched at render time; custom icons live in the shared
# assets/diagrams/ (Strands, Lambda coin, local-stack tools). Icon PNGs are
# re-rendered from their SVG sources at ICON_HEIGHT (needs librsvg's rsvg-convert).
# WIDTH sets output pixel width (0 = awsdac native; raise it for print).
AWSDAC_VERSION := v0.22.4
AWSDAC := $(shell go env GOPATH)/bin/awsdac
WIDTH ?= 0
ICON_HEIGHT ?= 500
diagrams:
	@test -x "$(AWSDAC)" || go install github.com/awslabs/diagram-as-code/cmd/awsdac@$(AWSDAC_VERSION)
	@for s in ../../assets/diagrams/icons/*.svg; do \
		rsvg-convert -h $(ICON_HEIGHT) "$$s" -o "../../assets/diagrams/icons/$$(basename $$s .svg).png"; \
	done
	@mkdir -p diagrams/png
	@for f in diagrams/yaml/*.yaml; do \
		echo "  $$f"; \
		"$(AWSDAC)" -f "$$f" -o "diagrams/png/$$(basename $$f .yaml).png" --width $(WIDTH); \
	done

test:
	uv run --with pytest --with httpx --with starlette --with uvicorn --with boto3 --with docker pytest tests/ -q

lint:
	uvx ruff format src/bureau-elt/*.py src/feature-store/*.py src/lake-basics/*.py
	uvx ruff check src/bureau-elt/*.py src/feature-store/*.py src/lake-basics/*.py
	uvx mbake format Makefile aws/Makefile
	uvx mbake validate Makefile aws/Makefile
	-uvx ty check .

features-lookup:
	uv run src/feature-store/lookup_features.py --applicant 10073

feature-api:
	# the local SageMaker Feature Store, both planes on :8007 (run in its own terminal)
	uv run local/sagemaker-local/app.py

feature-group:
	FEATURE_STORE_URL=http://localhost:8007 uv run src/feature-store/extra/feature_group_demo.py --applicant 10046

down:
	$(COMPOSE) down -v