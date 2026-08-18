.PHONY: up seed train train-challenger mlflow-train mlflow-ui sm-push sm-local sm-batch \
	amt lambda-build lambda-local lambda-registry-build lambda-registry-local fargate-build fargate-local pipeline pipeline-parity all down lint diagrams

# the whole local flow in one command, fully offline (no account): the local stack, the seed,
# both models trained in SageMaker local mode logged to the serverless MLflow, and the
# champion/challenger pipeline. For the native managed-MLflow run, use `make mlflow-train`
# (with credentials). make down when done.
all:
	$(MAKE) up
	$(MAKE) seed
	$(MAKE) mlflow-train SM_OFFLINE=1
	$(MAKE) pipeline

# Local cloud is S3Proxy (S3 API) only; MLflow is serverless (sqlite + S3Proxy locally). On AWS the one change is MLFLOW_TRACKING_URI -> a SageMaker MLflow App ARN.
export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

COMPOSE := docker compose -f local/mlflow-stack.yml

up:
	$(COMPOSE) up -d --wait

# synthesize the dataset and lay out the SageMaker training channels
seed:
	uv run data/generate_applications.py --rows 20000 --seed 7
	mkdir -p runs-local/input/data/train runs-local/input/data/validation
	cp data/split/train.csv data/split/feature_spec.json runs-local/input/data/train/
	cp data/split/test.csv runs-local/input/data/validation/

# train the incumbent scorecard in SageMaker local mode (Mode.LOCAL_CONTAINER): the SDK runs
# its custom container locally on the /opt/ml contract, not a bare docker run (no MLflow)
train:
	docker build -q -t ch02-scorecard:latest src/scorecard/
	rm -rf runs-local/model
	MODEL=scorecard uv run src/train_local.py

# train the monotone CatBoost challenger, likewise in SageMaker local mode
train-challenger:
	docker build -q -t ch02-challenger:latest src/challenger/
	rm -rf runs-local/model-challenger
	MODEL=challenger uv run src/train_local.py

# train both in SageMaker local mode (the container on your laptop, free compute), logged and
# registered to MLflow. Native (default): to the managed MLflow App (APP_ARN, resolved below
# from the ch02-mlflow app; real credentials), the same tracking server the cloud uses -- run
# experiments and register models without paying for training compute. SM_OFFLINE=1: to the
# serverless MLflow instead (sqlite + S3Proxy via `make up`), so it runs with no account;
# local mode has no writable mount but /opt/ml/model, so the sqlite store rides there and the
# driver threads it between the two runs into one mlflow-store/mlruns.db.
mlflow-train:
	docker build -q -t ch02-scorecard:latest src/scorecard/
	docker build -q -t ch02-challenger:latest src/challenger/
ifeq ($(SM_OFFLINE),1)
	$(MAKE) up
	rm -rf runs-local/model runs-local/model-challenger mlflow-store/mlruns.db
	MLFLOW=1 SM_OFFLINE=1 MODEL=scorecard uv run src/train_local.py
	MLFLOW=1 SM_OFFLINE=1 MODEL=challenger uv run src/train_local.py
else
	@test -n "$(APP_ARN)" && [ "$(APP_ARN)" != "None" ] || { echo "no ch02-mlflow MLflow App on this account: make -C aws mlflow-app (or pass APP_ARN=...); or run SM_OFFLINE=1 for the local sqlite stack"; exit 1; }
	rm -rf runs-local/model runs-local/model-challenger
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY MLFLOW=1 MLFLOW_TRACKING_URI="$(APP_ARN)" MODEL=scorecard uv run src/train_local.py
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY MLFLOW=1 MLFLOW_TRACKING_URI="$(APP_ARN)" MODEL=challenger uv run src/train_local.py
endif

# browse the runs on demand (no server runs otherwise)
mlflow-ui:
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 uvx mlflow ui \
		--backend-store-uri sqlite:///mlflow-store/mlruns.db --port 5000

# SageMaker local mode serving: the same BYOC image run through SDK v3 ModelBuilder, the
# container on this machine. The image is built locally (not in ECR), and ModelBuilder always
# pulls the serving image, so this runs in the no-account mode (SM_OFFLINE=1): the SDK's
# account, role, and image-pull calls are stubbed (src/serving/sagemaker_offline.py). The
# credentialed counterpart is the cloud endpoint (make -C aws deploy). Training differs --
# `make mlflow-train` runs natively, since ModelTrainer runs the local image without a pull.
# First `make seed train`. sm-push is the cloud image push and needs real credentials.
ACCOUNT ?= $(shell env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws sts get-caller-identity --query Account --output text)
ECR := $(ACCOUNT).dkr.ecr.$(AWS_DEFAULT_REGION).amazonaws.com

sm-push:
	aws ecr describe-repositories --repository-names ch02-scorecard --region $(AWS_DEFAULT_REGION) >/dev/null 2>&1 || \
		aws ecr create-repository --repository-name ch02-scorecard --region $(AWS_DEFAULT_REGION) >/dev/null
	aws ecr get-login-password --region $(AWS_DEFAULT_REGION) | docker login --username AWS --password-stdin $(ECR)
	docker buildx build --platform linux/amd64 --provenance=false -t $(ECR)/ch02-scorecard:latest --push src/scorecard

# real-time: the container as a local SageMaker endpoint (Mode.LOCAL_CONTAINER)
sm-local:
	docker build -q -t ch02-scorecard:latest src/scorecard/
	SM_OFFLINE=1 MODEL_IMAGE=ch02-scorecard:latest SAGEMAKER_ROLE_ARN=arn:aws:iam::000000000000:role/local \
		MODEL_PATH=runs-local/model uv run --no-project src/serving/local_endpoint.py

# batch: the same container as a local SageMaker Batch Transform job
sm-batch:
	docker build -q -t ch02-scorecard:latest src/scorecard/
	SM_OFFLINE=1 MODEL_IMAGE=ch02-scorecard:latest SAGEMAKER_ROLE_ARN=arn:aws:iam::000000000000:role/local \
		MODEL_PATH=runs-local/model uv run --no-project src/serving/local_transform.py

# Local AMT: Syne Tune runs the same search as managed AMT on your laptop (each trial a local subprocess, TPE scheduler, no instance, no Docker).
# Trials log to the serverless MLflow App, so this needs REAL AWS credentials: env -u strips the S3Proxy placeholders (override with make amt APP_ARN=...).
APP_ARN ?= $(shell env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws sagemaker list-mlflow-apps --query "Summaries[?Name=='ch02-mlflow'].Arn | [0]" --output text)

amt:
	@test -n "$(APP_ARN)" && test "$(APP_ARN)" != "None" || \
		{ echo "no MLflow App named ch02-mlflow on this account: make mlflow-app (aws/) or pass APP_ARN=..."; exit 1; }
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		MLFLOW_TRACKING_URI="$(APP_ARN)" uv run src/tuning/amt.py

# the local-tuning suite: trial contract, local/cloud search-space parity, and a
# two-trial TPE mini search, all offline, no Docker, no AWS
test:
	uv run --python 3.12 --with pytest --with syne-tune --with catboost \
		--with scikit-learn --with pandas --with numpy pytest tests/ -q

# assemble + build the Lambda container (model baked in), then run it via the RIE
lambda-build:
	cp src/challenger/challenger_model.py src/serving/lambda/challenger_model.py
	mkdir -p src/serving/lambda/model && cp runs-local/model-challenger/challenger.cbm runs-local/model-challenger/feature_spec.json src/serving/lambda/model/
	docker build -q -t ch02-lambda:latest src/serving/lambda/

lambda-local: lambda-build
	docker rm -f ch02-lambda-rie 2>/dev/null || true
	docker run -d --name ch02-lambda-rie -p 9010:8080 ch02-lambda:latest
	sleep 3
	curl -s -XPOST http://localhost:9010/2015-03-31/functions/function/invocations --data-binary @fixtures/sample.json; echo
	docker rm -f ch02-lambda-rie 2>/dev/null || true

# The registry-loading Lambda: no model baked in. mlflow-skinny + sagemaker-mlflow resolve
# the registered version and download its artifact at cold start (to /tmp). Tested offline
# against the local sqlite registry + S3Proxy; on AWS point MLFLOW_TRACKING_URI at the App ARN.
lambda-registry-build:
	cp src/challenger/challenger_model.py src/serving/lambda-registry/challenger_model.py
	docker build -q -t ch02-lambda-registry:latest src/serving/lambda-registry/

lambda-registry-local: lambda-registry-build
	docker rm -f ch02-lambda-registry-rie 2>/dev/null || true
	docker run -d --name ch02-lambda-registry-rie -p 9011:8080 \
		-e MLFLOW_TRACKING_URI=sqlite:////mlflow-store/mlruns.db \
		-e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:9000 \
		-e AWS_ACCESS_KEY_ID=local -e AWS_SECRET_ACCESS_KEY=localsecret -e AWS_DEFAULT_REGION=us-east-1 \
		-e REGISTERED_MODEL=credit-challenger \
		-v $(CURDIR)/mlflow-store:/mlflow-store \
		ch02-lambda-registry:latest
	sleep 4
	curl -s -XPOST http://localhost:9011/2015-03-31/functions/function/invocations --data-binary @fixtures/sample.json; echo
	docker rm -f ch02-lambda-registry-rie 2>/dev/null || true

# assemble + build the Fargate image, then run it as a local ECS task (compose)
fargate-build: train-challenger
	mkdir -p src/serving/fargate/model && cp runs-local/model-challenger/challenger.cbm runs-local/model-challenger/feature_spec.json src/serving/fargate/model/
	docker build -q --build-arg BASE_IMAGE=ch02-challenger:latest -t ch02-fargate:latest src/serving/fargate/

fargate-local: fargate-build
	cd src/serving/fargate && docker compose -f fargate-task.yml -f ecs-endpoints.yml up -d && sleep 5 && \
		curl -s -XPOST http://localhost:8093/invocations -H 'Content-Type: application/json' --data-binary @../../../fixtures/sample.json; echo; \
		docker compose -f fargate-task.yml -f ecs-endpoints.yml down

# the champion/challenger DAG as a local SageMaker Pipeline: every ProcessingStep runs as a
# local container (instance_type=local), staging to the local S3 (S3Proxy). Fully local --
# no real AWS: the host does the S3 I/O, AWS_ENDPOINT_URL_S3 points it at S3Proxy, and the
# dummy creds sign for it. Needs make up first.
SCORE_BUCKET ?= ch02-pipeline
pipeline:
	docker build -q -t ch02-step:local src/pipeline/step_image/
	@aws --endpoint-url http://localhost:9000 s3 ls s3://$(SCORE_BUCKET) >/dev/null 2>&1 \
		|| { echo "start the local S3 first: make up"; exit 1; }
	AWS_ENDPOINT_URL_S3=http://localhost:9000 PIPELINE_MODE=local STEP_IMAGE=ch02-step:local \
		SCORE_BUCKET=$(SCORE_BUCKET) SAGEMAKER_ROLE_ARN=arn:aws:iam::000000000000:role/local \
		SCORE_PREFIX=$(LOCAL_SCORES) \
		uv run src/pipeline/pipeline.py

# the pipeline parity check: the local run writes to S3Proxy, a cloud run to real S3; pull
# both and compare byte for byte. LOCAL from S3Proxy (dummy creds), AWS from real S3 (env -u,
# the cloud pipeline's SageMaker default bucket -- override AWS_SCORE_BUCKET if it differs).
LOCAL_SCORES ?= ch02/pipeline-scores-local
AWS_SCORES ?= ch02/pipeline-scores-aws
AWS_SCORE_BUCKET ?= sagemaker-$(AWS_DEFAULT_REGION)-$(ACCOUNT)
pipeline-parity:
	aws --endpoint-url http://localhost:9000 s3 cp s3://$(SCORE_BUCKET)/$(LOCAL_SCORES)/scores.csv /tmp/ch02-scores-local.csv
	@env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		aws s3 cp s3://$(AWS_SCORE_BUCKET)/$(AWS_SCORES)/scores.csv /tmp/ch02-scores-aws.csv
	@shasum -a 256 /tmp/ch02-scores-local.csv /tmp/ch02-scores-aws.csv
	@cmp /tmp/ch02-scores-local.csv /tmp/ch02-scores-aws.csv && echo "parity: scores.csv byte-identical"

down:
	$(COMPOSE) down
	docker rm -f ch02-lambda-rie 2>/dev/null || true

lint:
	uvx ruff check .

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