COMPOSE := docker compose -f local/batch-stack.yml

.PHONY: up seed run pipeline all events schedule query down clean lint diagrams

# the whole local flow in one command: the warehouse and local S3, the seed, and the batch
# pipeline in SageMaker local mode. make down when done.
all:
	$(MAKE) up
	$(MAKE) seed
	$(MAKE) pipeline

# The local stack sees dummy credentials (below); real-AWS targets shed them so
# boto3 falls through to the profile chain. Run those through $(REAL_AWS).
REAL_AWS := env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY

export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

up:
	$(COMPOSE) up -d --wait

seed:
	uv run data/generate_portfolio.py --rows 20000 --seed 5
	uv run src/steps/seed_warehouse.py

# the batch run, step by step: shortlist -> score -> decide -> apply
run:
	uv run src/steps/shortlist.py
	uv run src/steps/score.py
	uv run src/steps/decide.py
	uv run src/steps/apply_limits.py

# the local EventBridge and SNS, in its own terminal
events:
	uv run local/events-local/app.py

# the schedule and the alerts topic, same shapes as aws/template.yaml
schedule:
	uv run src/steps/schedule.py

# the same run as a local SageMaker Pipeline: every step (including the ModelTrainer
# training step) runs as a local container through the SDK's local control plane, staging to
# the local S3 (S3Proxy). Fully offline -- no account: the host does the S3 I/O
# (AWS_ENDPOINT_URL_S3), and SM_OFFLINE=1 has pipeline.py skip the SDK's IAM role check. The
# credentialed path is the managed AWS pipeline (make -C aws round). Needs make up + seed first.
BATCH_BUCKET ?= portfolio
pipeline:
	docker build -q -t ch05-step:local src/pipeline/step_image/
	SM_OFFLINE=1 AWS_ENDPOINT_URL_S3=http://localhost:9000 \
		PIPELINE_MODE=local STEP_IMAGE=ch05-step:local BATCH_BUCKET=$(BATCH_BUCKET) \
		SAGEMAKER_ROLE_ARN=arn:aws:iam::000000000000:role/local \
		WAREHOUSE_DSN=postgresql://analyst:analyst@host.docker.internal:5439/portfolio?sslmode=require \
		LAMBDA_WAREHOUSE_DSN=postgresql://analyst:analyst@localhost:5439/portfolio?sslmode=require \
		LAMBDA_HANDLERS='{"ch05-decide":"decide.handler","ch05-apply-limits":"handler.handler"}' \
		PYTHONPATH=src/pipeline:aws/lambda/apply_limits:aws/lambda/decide \
		S3_ENDPOINT=http://localhost:9000 SNS_ENDPOINT=http://localhost:8009 \
		TOPIC_ARN=arn:aws:sns:us-east-1:000000000000:ch05-batch-alerts \
		uv run --with psycopg2-binary src/pipeline/pipeline.py

query:
	$(COMPOSE) exec warehouse psql -h 127.0.0.1 -p 5433 -U analyst -d portfolio \
		-c "SELECT ROUND(AVG(current_limit)) AS avg_limit, COUNT(*) FROM customers"

down:
	$(COMPOSE) down -v

clean: down
	rm -rf data/portfolio.csv data/feature_spec.json catboost_info

lint:
	uvx ruff format data/*.py src/steps/*.py src/pipeline/pipeline.py src/pipeline/scripts/*.py local/events-local/*.py
	uvx ruff check data/*.py src/steps/*.py src/pipeline/pipeline.py src/pipeline/scripts/*.py
# Regenerate the architecture figures from diagrams/yaml/ with a pinned awsdac.
# Custom icons live in the shared assets/diagrams/; AWS icons fetched at render.
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