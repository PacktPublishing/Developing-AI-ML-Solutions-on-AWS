.PHONY: up faces enrol match match-impostor compare down diagrams lint

COMPOSE := docker compose -f local/vision-stack.yml
N ?= 40
SUBJECT ?= subject_000
FACES := data/generated/faces

# The local KYC vector-search loop: pgvector container, enrol the ID photos, then match a probe selfie (1:N).
up:
	$(COMPOSE) up -d --wait

# fetch real faces (UTKFace, CC0) and build id/selfie/impostor pairs (gitignored)
faces:
	set -a; . ../../.env; set +a; uv run etl/build_pairs.py --n $(N)

enrol:
	cd local && uv run --project . python enrol.py ../$(FACES)

# KYC verification: a genuine selfie's top hit is its own claimed subject -> matched
match:
	cd local && uv run --project . python match.py ../$(FACES)/probe/$(SUBJECT)/selfie.jpg --claim $(SUBJECT)

# an impostor selfie resolves to a different subject than claimed -> not matched
match-impostor:
	cd local && uv run --project . python match.py ../$(FACES)/impostors/$(SUBJECT)/selfie.jpg --claim $(SUBJECT)

# 1:1 compare + Captum explanation on a genuine pair
compare:
	cd local && uv run --project . python compare.py \
		../$(FACES)/enrolled/$(SUBJECT)/id.jpg ../$(FACES)/probe/$(SUBJECT)/selfie.jpg

down:
	$(COMPOSE) down -v

lint:
	uvx ruff check .

# Regenerate the architecture figures from diagrams/yaml/ with a pinned awsdac.
# WIDTH sets output pixel width (0 = awsdac native); ICON_HEIGHT re-renders custom SVG icons (needs rsvg-convert).
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