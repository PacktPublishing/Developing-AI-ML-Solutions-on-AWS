.PHONY: up seed run pipeline events schedule query down clean lint diagrams

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