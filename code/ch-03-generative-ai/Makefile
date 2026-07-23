.PHONY: up seed guardrail ask decide down lint test

# The element scripts live in sub-folders but import the shared seam (models.py,
# stores.py) from the chapter root, so the root is on the path.
export PYTHONPATH := $(CURDIR)

# Bedrock is the one cloud call and uses your real AWS credentials from the
# default chain. Set BEDROCK_LOCAL=1 to run the models on Ollama instead.
export AWS_DEFAULT_REGION ?= us-east-1
export BEDROCK_REGION ?= us-east-1

export PGHOST ?= localhost
export PGPORT ?= 5544
export PGUSER ?= underwriter
export PGPASSWORD ?= underwriter
export PGDATABASE ?= underwriting

Q ?= What DSCR floor applies to a solar project company?
DEAL ?= GreenRoof Installers, a residential solar installer, seeks a 3.5 million US dollar term loan for working capital.

up:
	docker compose up -d --wait

seed:
	uv run knowledge-base/corpus.py

guardrail:
	uv run underwriting-agent/guardrails.py create

ask:
	uv run underwriting-agent/agent.py ask --query "$(Q)" --guardrail

decide:
	uv run underwriting-agent/agent.py decide --deal "$(DEAL)"

down:
	docker compose down -v

lint:
	uvx ruff format *.py knowledge-base/*.py underwriting-agent/*.py
	uvx ruff check *.py knowledge-base/*.py underwriting-agent/*.py

# Unit tests plus a pgvector round-trip. The pgvector test provisions its own
# container with docker-py and stubs the embeddings, so it needs no cloud.
test:
	uv run --with pytest --with psycopg2-binary --with boto3 --with docker pytest tests/ -q