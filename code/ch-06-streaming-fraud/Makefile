.PHONY: all up seed train serve produce score firehose deliver lookup query diagrams down clean lint

# bootstrap the local stack in one command: the compose services, the synthetic stream, the
# trained scorer, and the serving container. The streaming demo itself is multi-terminal --
# run `make firehose` in its own terminal, then `make deliver`, `make produce`, and
# `make score`, then `make query`/`lookup`. make down when finished.
all: up seed train serve

# Custom icons live in assets/diagrams/. Icon PNGs re-render from SVG at
# ICON_HEIGHT (needs rsvg-convert); WIDTH sets pixel width (0 = awsdac native).
AWSDAC_VERSION := v0.22.4
AWSDAC := $(shell go env GOPATH)/bin/awsdac
WIDTH ?= 0
ICON_HEIGHT ?= 500

# Local credentials for the containerized services; any non-empty value works.
# On AWS, unset the *_ENDPOINT variables and use real credentials.
export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

COMPOSE := docker compose -f local/fraud-stack.yml

up:
	$(COMPOSE) up -d --wait

# synthesize the transaction stream, its labels, and the chronological split
seed:
	uv run data/generate_transactions.py --rows 50000 --seed 6

# fit the fraud classifier and freeze its operating threshold
train:
	uv run src/scoring/train.py

# serve the model locally (same image, same /ping,/invocations as the endpoint):
# SageMaker runs the image as `... serve` with the tarball at /opt/ml/model.
serve:
	docker build -q -t ch06-scoring:latest src/scoring/
	docker rm -f ch06-serve 2>/dev/null || true
	docker run -d --name ch06-serve -p 8080:8080 \
		-v $(PWD)/artifacts:/opt/ml/model:ro ch06-scoring:latest serve

# replay transactions onto the stream, as the card processor would
produce:
	uv run src/streaming/producer.py --limit 2000

# drain the stream: score each transaction, blocks to DynamoDB,
# every decision onward to the analytics stream
score:
	uv run src/streaming/consumer.py

# the local Firehose: serves the delivery-stream API and runs the
# buffer -> stage -> COPY loop; keep it in its own terminal
firehose:
	uv run local/firehose-local/app.py

# create the delivery stream: the same destination configuration
# aws/template.yaml declares, passed to the API instead
deliver:
	uv run src/streaming/downstream.py

# the processor's fast-path check: a block found stops the payment,
# an empty result lets it through. make lookup TX=tx-0000085
TX ?= tx-0000085
lookup:
	aws dynamodb get-item --endpoint-url http://localhost:8000 \
		--table-name blocked_transactions \
		--key '{"transaction_id": {"S": "$(TX)"}}' \
		--query 'Item.{transaction_id:transaction_id.S,score:score.N}'

# what the analyst sees in the warehouse once decisions are delivered
query:
	$(COMPOSE) exec warehouse psql -h 127.0.0.1 -p 5433 -U analyst -d fraud \
		-c "SELECT decision, COUNT(*), ROUND(AVG(score)::numeric, 4) AS avg_score \
		FROM fraud_decisions GROUP BY decision"

# seed the score-drift monitoring tables (reference, live scores, DBShap attribution)
# the Grafana board's monitoring row reads
monitor:
	uv run src/scoring/monitoring.py

# stop everything this chapter started: the compose stack and the serving
# container, which runs outside compose the way `make serve` started it
down:
	docker rm -f ch06-serve 2>/dev/null || true
	$(COMPOSE) down -v

# remove what the chapter generated: data, artifacts, training scratch, and
# the serving image. `make clean` returns the machine to its pre-chapter state.
clean: down
	rm -rf data/stream data/split data/labels.csv data/feature_spec.json \
		artifacts catboost_info
	docker rmi ch06-scoring:latest 2>/dev/null || true

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

lint:
	uvx ruff format data/*.py src/scoring/*.py src/streaming/*.py local/firehose-local/*.py
	uvx ruff check data/*.py src/scoring/*.py src/streaming/*.py local/firehose-local/*.py