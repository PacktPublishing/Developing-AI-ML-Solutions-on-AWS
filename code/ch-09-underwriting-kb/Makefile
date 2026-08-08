.PHONY: up gen seed ask cases down lint test

# The scripts under src/ and etl/ import the shared seam (models.py, stores.py)
# as top-level modules, so src/ is on the path.
SRC := $(CURDIR)/src
export PYTHONPATH := $(SRC)

COMPOSE := docker compose -f local/kb-stack.yml

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

# seed: embed the generated memos into pgvector
seed:
	uv run etl/embed_memos.py --memos data/generated/memos

ask:
	uv run src/retrieve.py ask --query "$(Q)"

cases:
	uv run src/retrieve.py cases --deal "$(DEAL)"

down:
	$(COMPOSE) down -v

lint:
	uvx ruff format etl/*.py src/*.py
	uvx ruff check etl/*.py src/*.py

# Unit test: a pgvector round-trip that generates a small corpus, embeds it with
# a deterministic stub, and searches it. Provisions its own container with
# docker-py and needs no cloud.
test:
	uv run --with pytest --with psycopg2-binary --with boto3 --with docker pytest tests/ -q