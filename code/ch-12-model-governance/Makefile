.PHONY: up apps score flip govern down lint test diagrams

COMPOSE := docker compose -f local/compose.yml

# bring up the AppConfig agent, the two chapter-2 models, and the flag-routing gateway
up:
	uv run etl/to_ion.py
	$(COMPOSE) up -d --build --wait
	@echo "gateway on :8080; champion + challenger behind the challenger_rollout flag (agent-evaluated)"

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