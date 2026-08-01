.PHONY: up seed run pipeline events schedule query down clean lint

include ../aws.mk

export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

up:
	docker compose up -d --wait

seed:
	uv run data/generate_portfolio.py --rows 20000 --seed 5
	uv run steps/seed_warehouse.py

# the batch run, step by step: shortlist -> score -> decide -> apply
run:
	uv run steps/shortlist.py
	uv run steps/score.py
	uv run steps/decide.py
	uv run steps/apply_limits.py

# the local EventBridge and SNS, in its own terminal
events:
	uv run events-local/app.py

# the schedule and the alerts topic, same shapes as aws/template.yaml
schedule:
	uv run steps/schedule.py

# the same run as a SageMaker Pipeline: local executor (Docker), real
# control plane — real credentials, a real S3 bucket for step I/O, the
# local warehouse. Needs make up + seed and BATCH_BUCKET + SAGEMAKER_ROLE_ARN.
pipeline:
	docker build -q -t ch05-step:local pipeline/step_image/
	PIPELINE_MODE=local STEP_IMAGE=ch05-step:local \
		WAREHOUSE_DSN=postgresql://analyst:analyst@host.docker.internal:5439/portfolio \
		$(REAL_AWS) uv run pipeline/pipeline.py

query:
	docker compose exec warehouse psql -h 127.0.0.1 -p 5433 -U analyst -d portfolio \
		-c "SELECT ROUND(AVG(current_limit)) AS avg_limit, COUNT(*) FROM customers"

down:
	docker compose down -v

clean: down
	rm -rf data/portfolio.csv data/feature_spec.json catboost_info

lint:
	uvx ruff format data/*.py steps/*.py pipeline/pipeline.py pipeline/scripts/*.py events-local/*.py
	uvx ruff check data/*.py steps/*.py pipeline/pipeline.py pipeline/scripts/*.py