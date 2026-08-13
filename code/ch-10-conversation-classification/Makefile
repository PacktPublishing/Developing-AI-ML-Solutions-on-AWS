.PHONY: up prepare classify route eval all down lint test diagrams

# the whole local flow in one command: the local S3, the datasets, the batch classification
# (local Ollama), the routing, and the evaluation. make down when done.
all:
	$(MAKE) up
	$(MAKE) prepare
	$(MAKE) classify
	$(MAKE) route
	$(MAKE) eval

# src/ holds the shared seam (models.py, batch.py); put it on the path.
export PYTHONPATH := $(CURDIR)/src

COMPOSE := docker compose -f local/batch-stack.yml
BUCKET ?= ch10-batch
DATASET ?= conversations
# prompt variant: a (bare category names) or b (defined category borders)
VARIANT ?= a

# Local by default: Bedrock is the Ollama-backed batch shim, S3 is S3Proxy.
export BEDROCK_LOCAL ?= 1
export S3_ENDPOINT ?= http://localhost:9000
export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret

up:
	$(COMPOSE) up -d --wait
	@aws --endpoint-url $(S3_ENDPOINT) s3 mb s3://$(BUCKET) 2>/dev/null || true
	@echo "s3proxy up; bucket $(BUCKET) ready"

# prepare: build the labeled datasets (RetailBanking conversations + banking77).
# RetailBanking is gated on Hugging Face, so pass HF_TOKEN=... the first time.
prepare:
	HF_TOKEN=$(HF_TOKEN) uv run etl/prepare.py --out data/generated

# classify: run the batch inference job (local shim here) and score it
classify:
	uv run src/classify.py --dataset $(DATASET) --bucket $(BUCKET) --variant $(VARIANT) $(if $(LIMIT),--limit $(LIMIT),)

# route: publish each classified conversation to its team's topic (from-source SNS)
route:
	uv run src/route.py --dataset $(DATASET)

# eval: accuracy plus multiclass Brier and log loss against the labels
eval:
	uv run src/evaluate.py --dataset $(DATASET)

down:
	$(COMPOSE) down -v

lint:
	uvx ruff format etl/*.py src/*.py
	uvx ruff check etl/*.py src/*.py

test:
	uv run --with pytest --with boto3 --with ollama pytest tests/ -q

# Regenerate the architecture figures from diagrams/yaml/ with a pinned awsdac (AWS icons fetched at render time, custom icons from shared assets/diagrams/).
# Icon PNGs are re-rendered from SVG at ICON_HEIGHT (needs rsvg-convert); WIDTH sets output pixel width (0 = awsdac native).
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