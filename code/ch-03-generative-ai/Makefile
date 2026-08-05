.PHONY: up seed guardrail ask decide serve down lint test diagrams

# The element scripts live under src/ but import the shared seam (models.py,
# stores.py) as top-level modules, so src/ is on the path.
SRC := $(CURDIR)/src
export PYTHONPATH := $(SRC)

COMPOSE := docker compose -f local/rag-stack.yml

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
	$(COMPOSE) up -d --wait

seed:
	uv run src/knowledge-base/corpus.py

# BEDROCK_LOCAL=1 exercises the local shim (no AWS); unset creates the managed
# Bedrock guardrail on your account.
guardrail:
	uv run src/underwriting-agent/guardrails.py $(if $(BEDROCK_LOCAL),local,create)

ask:
	uv run src/underwriting-agent/agent.py ask --query "$(Q)" --guardrail

decide:
	uv run src/underwriting-agent/agent.py decide --deal "$(DEAL)"

# serve the agent on the AgentCore contract (POST /invocations on :8080);
# same request shape locally and on the managed runtime
serve:
	uv run src/underwriting-agent/agent.py serve

down:
	$(COMPOSE) down -v

lint:
	uvx ruff format src/*.py src/knowledge-base/*.py src/underwriting-agent/*.py
	uvx ruff check src/*.py src/knowledge-base/*.py src/underwriting-agent/*.py

# Unit tests plus a pgvector round-trip. The pgvector test provisions its own
# container with docker-py and stubs the embeddings, so it needs no cloud.
test:
	uv run --with pytest --with psycopg2-binary --with boto3 --with docker pytest tests/ -q

# Regenerate the architecture figures from diagrams/yaml/ with a pinned awsdac (AWS diagram-as-code).
# ICON_HEIGHT sizes the custom-icon PNGs (needs rsvg-convert); WIDTH sets output pixel width (0 = awsdac native).
AWSDAC_VERSION := v0.22.4
AWSDAC := $(shell go env GOPATH)/bin/awsdac
WIDTH ?= 0
ICON_HEIGHT ?= 500
diagrams:
	@test -x "$(AWSDAC)" || go install github.com/awslabs/diagram-as-code/cmd/awsdac@$(AWSDAC_VERSION)
	@for s in ../../assets/diagrams/icons/*.svg; do \
		rsvg-convert -h $(ICON_HEIGHT) "$$s" -o "../../assets/diagrams/icons/$$(basename $$s .svg).png"; \
	done
	@for f in diagrams/yaml/*.yaml; do \
		echo "  $$f"; \
		"$(AWSDAC)" -f "$$f" -o "diagrams/png/$$(basename $$f .yaml).png" --width $(WIDTH); \
	done