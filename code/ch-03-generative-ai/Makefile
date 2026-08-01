.PHONY: up seed guardrail ask decide serve down lint test diagrams

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

# serve the agent on the AgentCore contract (POST /invocations on :8080);
# same request shape locally and on the managed runtime
serve:
	uv run underwriting-agent/agent.py serve

down:
	docker compose down -v

lint:
	uvx ruff format *.py knowledge-base/*.py underwriting-agent/*.py
	uvx ruff check *.py knowledge-base/*.py underwriting-agent/*.py

# Unit tests plus a pgvector round-trip. The pgvector test provisions its own
# container with docker-py and stubs the embeddings, so it needs no cloud.
test:
	uv run --with pytest --with psycopg2-binary --with boto3 --with docker pytest tests/ -q

# Regenerate the architecture figures from diagrams/yaml/ with a pinned awsdac
# (AWS diagram-as-code). AWS service icons are fetched from AWS at render time;
# the custom icons (Strands, Lambda coin) are vendored under ../../assets/diagrams/icons/ as
# SVG sources and declared in diagrams/defs/custom-icons.yaml, so the figures
# reproduce from the repo. The icon PNGs awsdac consumes are (re)rendered from
# those SVGs at ICON_HEIGHT px (needs librsvg's rsvg-convert). WIDTH sets output
# pixel width (0 = awsdac native; raise it for print — awsdac stamps 72 dpi in
# metadata regardless, so pixels are what matter).
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