.PHONY: up sm-local apps score flip govern all down lint test diagrams

COMPOSE := docker compose -f local/rollout-stack.yml

# the whole local flow in one command: the rollout stack (AppConfig agent + both models +
# gateway), a batch of applications, the champion/challenger split, and the auditor's
# governance record. make down when done.
all:
	$(MAKE) up
	$(MAKE) apps
	$(MAKE) score
	$(MAKE) govern

# Option 1, the rollout stack: the AppConfig agent, the two chapter-2 models, and the
# flag-routing gateway, all on one compose network so `make flip` widens with nothing
# restarted. This is the champion/challenger A/B end to end.
up:
	uv run etl/to_ion.py
	$(COMPOSE) up -d --build --wait
	@echo "gateway on :8080; champion + challenger behind the challenger_rollout flag (agent-evaluated)"

# Option 2, SageMaker local mode: serve one model through the same ModelBuilder path the
# cloud uses (Mode.LOCAL_CONTAINER) rather than the compose stack; swapping the Mode is the
# only change to reach the serverless endpoint. Needs AWS creds, a SageMaker role (SM_ROLE),
# and the model image in ECR (make -C aws deploy builds it). MODEL=champion|challenger.
REGION ?= us-east-1
# Lazy (=) so only the sm-local target resolves the account, not every make invocation.
ACCOUNT = $(shell aws sts get-caller-identity --query Account --output text)
ECR = $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com
SM_ROLE ?=
MODEL ?= champion
sm-local:
	@test -n "$(SM_ROLE)" || { echo "set SM_ROLE=<sagemaker exec role arn>"; exit 1; }
	@case "$(MODEL)" in \
		champion) MP=../ch-02-mlops/runs-local/model ;; \
		challenger) MP=../ch-02-mlops/runs-local/model-challenger ;; \
		*) echo "MODEL must be champion or challenger"; exit 1 ;; \
	esac; \
	MODEL_IMAGE=$(ECR)/ch12-$(MODEL):latest SAGEMAKER_ROLE_ARN=$(SM_ROLE) MODEL_PATH=$$MP \
	  uv run --no-project src/serve_local.py

# a batch of applications to route (reuses chapter 2's held-out set)
apps:
	uv run etl/make_applications.py --n 300 --out data/applications.jsonl

# route the batch through the gateway and show the champion/challenger split
score:
	uv run etl/score.py data/applications.jsonl

# widen the rollout, e.g. make flip PCT=50. Locally the agent reads the Ion file once at
# startup, so we nudge the sidecar to re-read it; the gateway and models never restart.
# On AWS the agent polls the AppConfig service, so a deployed flag needs no nudge at all.
flip:
	uv run etl/flip.py $(PCT)
	uv run etl/to_ion.py
	$(COMPOSE) restart appconfig-agent

# the auditor's record: model cards, the registry, and lineage from governance/models.json
govern:
	uv run src/governance.py --models governance/models.json --out outputs/governance

down:
	$(COMPOSE) down

lint:
	uvx ruff check .

# the rollout + gateway unit tests
test:
	uv run --with pytest --with boto3 --with fastapi --with httpx pytest tests/ -q

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