.PHONY: data train monitor run pipeline all up down lint test diagrams clean

# the whole local flow in one command: synthesize the batches, train the scorecard in
# SageMaker local mode, bring up the local S3, run the monitor, and the pipeline. make down when done.
all:
	$(MAKE) data
	$(MAKE) train
	$(MAKE) up
	$(MAKE) run
	$(MAKE) pipeline

export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

# The local S3 (S3Proxy) that stands in for the artifact bucket; make run and make
# pipeline both stage to it, so the local pipeline needs no real AWS at all.
COMPOSE := docker compose -f local/monitoring-stack.yml
S3_ENDPOINT := http://localhost:9000
S3_LOCAL := aws --endpoint-url $(S3_ENDPOINT)

IMAGE := ch11-scorecard:latest

# synthesize the reference and current (drifted) batches, then lay out the training channel
# (the reference batch is the channel the scorecard trains on)
data:
	uv run etl/make_batches.py
	mkdir -p runs-local/input/data/reference
	cp data/generated/reference.csv runs-local/input/data/reference/

# train the scorecard in SageMaker local mode: the SDK's ModelTrainer (Mode.LOCAL_CONTAINER)
# runs the same image the pipeline and the AWS jobs use, on the /opt/ml contract, as a local
# container rather than a bare docker run. Produces runs-local/model/scorecard.cbm.
train:
	docker build -q -t $(IMAGE) .
	rm -rf runs-local/model
	uv run src/train_local.py

# the drift monitors: PSI on scores and features, SHAP attribution NDCG
monitor:
	PYTHONPATH=src uv run src/monitor.py --model runs-local/model/scorecard.cbm

# up/down: the local artifact store (S3Proxy). Bring it up before make run to land the
# monitoring artifacts in a real S3 bucket locally, as the Processing step does on AWS.
up:
	$(COMPOSE) up -d
	@echo "local S3 (S3Proxy) on :9000, bucket s3://ch11-monitoring"
down:
	$(COMPOSE) down -v

# the monitoring job: SageMaker-shaped artifacts plus CloudWatch metrics and alarms (local
# shim). When the local S3 store is up (make up), upload the artifacts to it, standing in
# for SageMaker uploading /opt/ml/processing/output to S3.
run:
	MONITOR_LOCAL=1 PYTHONPATH=src uv run src/run_monitor.py --model runs-local/model/scorecard.cbm
	@if $(S3_LOCAL) s3 ls s3://ch11-monitoring >/dev/null 2>&1; then \
		$(S3_LOCAL) s3 sync outputs/monitoring s3://ch11-monitoring/monitoring; \
		echo "artifacts uploaded to s3://ch11-monitoring/monitoring (local S3)"; \
	fi

# the same run as a local SageMaker Pipeline: the Transform and Processing containers
# run here (instance_type=local, the SDK's local control plane). In local mode the host
# does the S3 staging, so pointing boto3 at S3Proxy (AWS_ENDPOINT_URL_S3) keeps it fully
# local -- no real AWS. Needs make up + make data + make train first.
ARTIFACT_BUCKET ?= ch11-monitoring
pipeline:
	docker build -q -t $(IMAGE) .
	@$(S3_LOCAL) s3 ls s3://$(ARTIFACT_BUCKET) >/dev/null 2>&1 \
		|| { echo "start the local S3 first: make up"; exit 1; }
	AWS_ENDPOINT_URL_S3=$(S3_ENDPOINT) PIPELINE_MODE=local IMAGE_URI=$(IMAGE) \
		SAGEMAKER_ROLE_ARN=arn:aws:iam::000000000000:role/local \
		ARTIFACT_BUCKET=$(ARTIFACT_BUCKET) DATA_DIR=data/generated MODEL_DIR=runs-local/model \
		uv run aws/pipeline.py

lint:
	uvx ruff check .

# the PSIDetector unit tests: quantile vs model-border bins, and JSON round-trip
test:
	uv run --with pytest --with catboost --with pandas --with numpy pytest tests/ -q

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

clean:
	rm -rf runs-local outputs data/generated catboost_info