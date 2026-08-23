.PHONY: test faces-sheet up faces register match match-fraudster compare bench serve-inprocess all down diagrams lint

# the whole local flow in one command: the pgvector store, the synthetic faces, an registration,
# and a genuine match. Then try match-fraudster / compare. make down when done.
all: up faces register match

COMPOSE := docker compose -f local/vision-stack.yml
N ?= 40
SUBJECT ?= subject_000
FACES := data/generated/faces

# The local KYC vector-search loop: pgvector container, register the ID photos, then match a probe selfie (1:N).
up:
	$(COMPOSE) up -d --wait

# fetch real faces (UTKFace, CC0) and build id/selfie/fraudster pairs (gitignored)
faces:
	set -a; . ../../.env; set +a; uv run etl/build_pairs.py --n $(N)

register:
	cd local && uv run --project . python register.py ../$(FACES)

# KYC verification: a genuine selfie's top hit is its own claimed subject -> matched
match:
	cd local && uv run --project . python match.py ../$(FACES)/probe/$(SUBJECT)/selfie.jpg --claim $(SUBJECT)

# an fraudster selfie resolves to a different subject than claimed -> not matched
match-fraudster:
	cd local && uv run --project . python match.py ../$(FACES)/fraudsters/$(SUBJECT)/selfie.jpg --claim $(SUBJECT)

# the embedder behind the SageMaker inference contract, in process, on the Apple GPU
serve-inprocess:
	SM_OFFLINE=1 PYTHONPATH=src FACES_DIR=data/generated/faces uv run local/serve_inprocess.py

# CPU vs Apple GPU (MPS) for the two embedding stages, over the registered photos
bench:
	cd local && uv run --project . python bench_device.py -n $(N)

# contact sheet of the test set: registered ID, genuine probe, fraudster, per subject
faces-sheet:
	cd local && uv run --project . python faces_sheet.py ../$(FACES) -n 3

# 1:1 compare + Captum explanation on a genuine pair
compare:
	cd local && uv run --project . python compare.py \
		../$(FACES)/registered/$(SUBJECT)/id.jpg ../$(FACES)/probe/$(SUBJECT)/selfie.jpg

down:
	$(COMPOSE) down -v

lint:
	uvx ruff check .

# the vector store against a throwaway pgvector on :5508, so `make up` can stay
# running on :5507; the suite provisions its own container through docker-py
test:
	uv run --with pytest --with "psycopg[binary]" --with psycopg-pool --with docker pytest tests/ -q

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