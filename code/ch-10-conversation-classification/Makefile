.PHONY: up prepare classify decide serve-decider serve-agent sm-local sm-local-image route eval all down lint test diagrams

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
DECIDER_CHECKPOINT ?= StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
# prompt variant: a (bare category names) or b (defined category borders).
# classify also takes c (borders plus few-shot); decide takes a or b.
VARIANT ?= a

# Local by default: Bedrock is the Ollama-backed batch shim, S3 is S3Proxy.
export BEDROCK_LOCAL ?= 1
export S3_ENDPOINT ?= http://localhost:9000
# Only the S3Proxy targets need placeholder credentials. Cloud callers use
# the normal AWS credential chain, including the selected profile.
up classify: export AWS_ACCESS_KEY_ID ?= local
up classify: export AWS_SECRET_ACCESS_KEY ?= localsecret

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

# decide: route the same conversations with a decision model rather than an LLM.
# local loads the model here; http points at any System One server (Strands
# Decider's or Kev's, they serve the same endpoint); agentcore needs
# AGENT_RUNTIME_ARN. THRESHOLD is the confidence below which a conversation is
# left for a person.
export DECIDER_BACKEND ?= local
export DECIDER_URL ?=
# /invocations instead of /v1/systemone runs the local round through the exact
# path AgentCore Runtime invokes, which Strands Decider's server also serves.
export DECIDER_PATH ?= /v1/systemone
THRESHOLD ?= 0
decide:
	uv run --with strands-decider src/decide.py --dataset $(DATASET) \
		--variant $(VARIANT) --threshold $(THRESHOLD) $(if $(LIMIT),--limit $(LIMIT),)

# serve-decider: the baseline model behind an HTTP server, so `make decide
# DECIDER_BACKEND=http DECIDER_URL=http://127.0.0.1:8009` reaches it the same
# way it reaches Kev or AgentCore.
# 8009 is a common default for these servers and collides easily; override with
# PORT= if 8019 is taken too.
PORT ?= 8019
serve-decider:
	uv run --with strands-decider strands-decider serve $(DECIDER_CHECKPOINT) --port $(PORT)

# serve-agent: the routing agent behind the AgentCore Runtime contract, the file
# that deploys. It asks the model rather than holding it, so it needs a
# deployed endpoint to be useful:
#   make serve-agent ENDPOINT_NAME=ch10-decider
#   curl -X POST localhost:8080/invocations -H 'content-type: application/json' \
#     -d '{"text": "my debit card was declined", "threshold": 0.6}'
AGENT_PORT ?= 8080
ENDPOINT_NAME ?= ch10-decider
serve-agent:
	PORT=$(AGENT_PORT) ENDPOINT_NAME=$(ENDPOINT_NAME) \
		uv run --with bedrock-agentcore --with boto3 --with "botocore[crt]" agent/main.py

# sm-local: the model's container as a SageMaker endpoint in local mode, which
# starts it the way the service does (`docker run <image> serve`) rather than
# the way a person would. A plain docker run uses CMD and passes even when the
# image has no serve on PATH, which is how that failure once reached a deployed
# endpoint. Needs AWS credentials and any role ARN; local mode never assumes it.
#
# The CPU variant of the base image, because a laptop has no NVIDIA GPU.
SM_LOCAL_IMAGE ?= ch10-decider:cpu
# Not the distribution image: it is amd64 only, so on Apple silicon it builds
# emulated and takes hours. This one is multi-arch and builds natively.
SM_LOCAL_BASE ?= ghcr.io/astral-sh/uv:python3.12-bookworm-slim
SM_LOCAL_TORCH ?= https://download.pytorch.org/whl/cpu
SAGEMAKER_ROLE_ARN ?= arn:aws:iam::000000000000:role/local-mode-placeholder
sm-local-image:
	@# No --platform: this builds for whatever this machine is, natively. The
	@# weights are baked in here as they are for the endpoint, so the download
	@# happens once at build time instead of on every run.
	docker build --build-arg BASE_IMAGE=$(SM_LOCAL_BASE) \
		--build-arg TORCH_INDEX=$(SM_LOCAL_TORCH) \
		-t $(SM_LOCAL_IMAGE) model/

sm-local: sm-local-image
	MODEL_IMAGE=$(SM_LOCAL_IMAGE) DEVICE=cpu \
		PYTHONPATH=src uv run src/serve_local.py

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
# The shared icons come from `make icons` at the repository root; WIDTH sets
# output pixel width (0 = awsdac native).
AWSDAC_VERSION := v0.22.4
AWSDAC := $(shell go env GOPATH)/bin/awsdac
WIDTH ?= 0
diagrams:
	@test -x "$(AWSDAC)" || go install github.com/awslabs/diagram-as-code/cmd/awsdac@$(AWSDAC_VERSION)
	@$(MAKE) -s -C ../.. icons
	@mkdir -p diagrams/png
	@for f in diagrams/yaml/*.yaml; do \
		echo "  $$f"; \
		"$(AWSDAC)" -f "$$f" -o "diagrams/png/$$(basename $$f .yaml).png" --width $(WIDTH); \
	done
