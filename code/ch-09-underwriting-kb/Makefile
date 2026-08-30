.PHONY: neural up gen seed ask cases app all down lint test diagrams

# the whole local flow in one command: the vector store, the synthetic corpus, the embeddings,
# and a case-retrieval query. make down when done.
all:
	$(MAKE) up
	$(MAKE) gen
	$(MAKE) seed
	$(MAKE) cases

# The scripts under src/ and etl/ import the shared seam (models.py, stores.py)
# as top-level modules, so src/ is on the path.
SRC := $(CURDIR)/src
export PYTHONPATH := $(SRC)

COMPOSE := docker compose -f local/kb-stack.yml

# The OpenSearch node and Dashboards sit behind a compose
# profile named after the store, started only when that store is selected.

export S3VECTORS_LOCAL ?= $(BEDROCK_LOCAL)
export S3_ENDPOINT ?= http://localhost:9000

# Bedrock is the one cloud call and uses your real AWS credentials from the
# default chain. Set BEDROCK_LOCAL=1 to run the models on Ollama instead.
export AWS_DEFAULT_REGION ?= us-east-1
export BEDROCK_REGION ?= us-east-1

export PGHOST ?= localhost
export PGPORT ?= 5545
export PGUSER ?= underwriter
export PGPASSWORD ?= underwriter
export PGDATABASE ?= underwriting

COUNT ?= 200
SEED ?= 7
Q ?= How is DTI assessed for a grocery business?
DEAL ?= A logistics firm seeks a 500,000 US dollar working-capital loan for six months.

up:
	$(COMPOSE) up -d --wait

# gen: write the synthetic memo corpus (reproducible from the seed)
gen:
	uv run etl/gen_memos.py --out data/generated/memos --count $(COUNT) --seed $(SEED) --messy

# seed: embed the generated memos into the OpenSearch index
seed:
	uv run etl/embed_memos.py --memos data/generated/memos

ask:
	uv run src/retrieve.py ask --query "$(Q)"

cases:
	uv run src/retrieve.py cases --deal "$(DEAL)"

# app: the FastAPI ask/recommend UI over the selected store, on http://localhost:8080.
# The same app image deploys to Lambda; here it runs under uvicorn.
app:
	uv run app/main.py

down:
	$(COMPOSE) down -v

lint:
	uvx ruff format etl/*.py src/*.py
	uvx ruff check etl/*.py src/*.py

# Unit test: an OpenSearch round-trip that generates a small corpus, embeds it with
# a deterministic stub, and searches it. Provisions its own container with
# docker-py and needs no cloud.
test:
	uv run --with pytest --with boto3 --with opensearch-py pytest tests/ -q

# neural: let the cluster embed instead of embedding first. Registers a pretrained
# model inside OpenSearch, indexes memo text through an ingest pipeline, and runs a
# neural query that carries text rather than a vector. Needs `make up` first.
neural:
	uv run --no-project local/neural_search.py

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