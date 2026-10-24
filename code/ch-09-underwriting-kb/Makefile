.PHONY: neural up gen seed ask cases app all down lint test diagrams statements review eval auth call

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

# The underwriters' user pool on the local Cognito (local/cognito): fixed ids, so the
# app, the page, and underwriter.py agree without reading them back. The password
# is local-only and meets the template's policy (12 characters, upper, lower,
# number, symbol).
export COGNITO_ENDPOINT := http://localhost:9229
export COGNITO_POOL_ID := us-east-1_ch09Local
export COGNITO_CLIENT_ID := ch09underwriterapplocal000
export UNDERWRITER_PASSWORD ?= Local-underwriter-1!

COUNT ?= 200
SEED ?= 7
Q ?= How is DTI assessed for a grocery business?
DEAL ?= A logistics firm seeks a 500,000 US dollar working-capital loan for six months.

up:
	$(COMPOSE) up -d --build --wait

# statements: the synthetic bank statement PDFs an underwriter uploads for review
statements:
	uv run etl/gen_statements.py --out data/generated/statements --count 12 --seed 11

# review: upload one statement and let the agent review it against the archive
CASE ?= CASE-20260000
review: statements
	uv run --with pdfplumber src/review.py --case $(CASE)

# gen: write the synthetic memo corpus (reproducible from the seed)
gen:
	uv run etl/gen_memos.py --out data/generated/memos --count $(COUNT) --seed $(SEED) --messy

# seed: embed the generated memos into the OpenSearch index
seed:
	uv run etl/embed_memos.py --memos data/generated/memos --policy data/policy

# eval: how often the right memo or policy is in the top 5, k-NN alone against hybrid
eval:
	uv run etl/evaluate.py --memos data/generated/memos

ask:
	uv run src/retrieve.py ask --query "$(Q)"

cases:
	uv run src/retrieve.py cases --deal "$(DEAL)"

# app: the FastAPI ask/recommend UI over the selected store, on http://localhost:8080.
# The same app image deploys to Lambda; here it runs under uvicorn.
# The app requires a signed-in underwriter (`make auth` creates one).
app:
	COGNITO_ISSUER=$(COGNITO_ENDPOINT)/$(COGNITO_POOL_ID) uv run app/main.py

# auth: the underwriter in the local pool, with a permanent password
auth:
	uv run --no-project aws/underwriter.py add

# call: POST to the local app as the underwriter, or without a token (NO_TOKEN=1)
ROUTE ?= ask
BODY ?= {"query": "$(Q)"}
call:
	uv run --no-project aws/underwriter.py call $(ROUTE) '$(BODY)' $(if $(NO_TOKEN),--no-token)

down:
	$(COMPOSE) down -v

lint:
	uvx ruff format etl/*.py src/*.py app/*.py aws/*.py
	uvx ruff check etl/*.py src/*.py app/*.py aws/*.py

# Unit test: an OpenSearch round-trip that generates a small corpus, embeds it with
# a deterministic stub, and searches it. Provisions its own container with
# docker-py and needs no cloud.
test:
	uv run --with pytest --with boto3 --with opensearch-py --with pdfplumber \
		--with reportlab --with "pyjwt[crypto]" --with pyyaml --with cryptography pytest tests/ -q

# neural: let the cluster embed instead of embedding first. Registers a pretrained
# model inside OpenSearch, indexes memo text through an ingest pipeline, and runs a
# neural query that carries text rather than a vector. Needs `make up` first.
neural:
	uv run --no-project local/neural_search.py

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