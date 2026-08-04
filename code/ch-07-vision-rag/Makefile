.PHONY: up faces enrol match match-impostor compare down diagrams lint

COMPOSE := docker compose -f local/vision-stack.yml
N ?= 40
SUBJECT ?= subject_000
FACES := data/generated/faces

# The local KYC vector-search loop: a pgvector container, enrol the ID photos,
# then match a probe selfie (1:N). Real faces + simulated selfies come from etl/.
# On AWS the same code runs on RDS pgvector behind the enrol/match Lambdas.
up:
	$(COMPOSE) up -d --wait

# fetch real faces (UTKFace, CC0) and build id/selfie/impostor pairs (gitignored)
faces:
	set -a; . ../../.env; set +a; uv run etl/build_pairs.py --n $(N)

enrol:
	cd local && uv run --project . python enrol.py ../$(FACES)

# 1:N match — a genuine selfie should hit its own subject above the threshold
match:
	cd local && uv run --project . python match.py ../$(FACES)/probe/$(SUBJECT)/selfie.jpg

# 1:N with an impostor selfie — the top score should fall below the threshold
match-impostor:
	cd local && uv run --project . python match.py ../$(FACES)/impostors/$(SUBJECT)/selfie.jpg

# 1:1 compare + Captum explanation on a genuine pair
compare:
	cd local && uv run --project . python compare.py \
		../$(FACES)/enrolled/$(SUBJECT)/id.jpg ../$(FACES)/probe/$(SUBJECT)/selfie.jpg

down:
	$(COMPOSE) down -v

lint:
	uvx ruff check .

# Regenerate the architecture figures from diagrams/yaml/ with a pinned awsdac.
# AWS icons are fetched at render time; custom icons live in the shared
# assets/diagrams/ (Strands, Lambda coin, local-stack tools). Icon PNGs are
# re-rendered from their SVG sources at ICON_HEIGHT (needs librsvg's rsvg-convert).
# WIDTH sets output pixel width (0 = awsdac native; raise it for print).
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