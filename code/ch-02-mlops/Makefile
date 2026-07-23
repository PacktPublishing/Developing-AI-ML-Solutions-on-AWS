.PHONY: up seed train train-challenger mlflow-train mlflow-ui serve score batch \
	amt lambda-build lambda-local fargate-build fargate-local pipeline-parity down lint

# The local cloud is S3Proxy (S3 API) only; MLflow is serverless — no tracking
# server. Locally it writes runs to a sqlite file and artifacts to S3Proxy; on
# AWS the one change is MLFLOW_TRACKING_URI -> a SageMaker MLflow App ARN.
export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

# /opt/ml training channels and the serverless-MLflow env, passed to the container.
INPUT := -v $(PWD)/runs-local/input:/opt/ml/input:ro -v $(PWD)/runs-local/output:/opt/ml/output
MLFLOW := -e MLFLOW_TRACKING_URI=sqlite:////mlflow-store/mlruns.db \
	-e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:9000 \
	-e MLFLOW_ARTIFACT_ROOT=s3://mlflow-artifacts \
	-e AWS_ACCESS_KEY_ID=local -e AWS_SECRET_ACCESS_KEY=localsecret -e AWS_DEFAULT_REGION=us-east-1 \
	-v $(PWD)/mlflow-store:/mlflow-store

up:
	docker compose up -d --wait

# synthesize the dataset and lay out the SageMaker training channels
seed:
	uv run data/generate_applications.py --rows 20000 --seed 7
	mkdir -p runs-local/input/data/train runs-local/input/data/validation runs-local/input/config runs-local/output
	cp data/split/train.csv data/split/feature_spec.json runs-local/input/data/train/
	cp data/split/test.csv runs-local/input/data/validation/
	printf '{"C":"1.0","max_iter":"1000","monotonic":"true"}' > runs-local/input/config/hyperparameters.json

# train the incumbent scorecard in its custom container (no MLflow)
train:
	docker build -q -t ch02-scorecard:latest scorecard/
	rm -rf runs-local/model && mkdir -p runs-local/model
	docker run --rm $(INPUT) -v $(PWD)/runs-local/model:/opt/ml/model ch02-scorecard:latest train

# train the monotone XGBoost challenger
train-challenger:
	docker build -q -t ch02-challenger:latest challenger/
	rm -rf runs-local/model-challenger && mkdir -p runs-local/model-challenger
	docker run --rm $(INPUT) -v $(PWD)/runs-local/model-challenger:/opt/ml/model ch02-challenger:latest train

# train both, logged to serverless MLflow (sqlite + S3Proxy); needs `make up`
mlflow-train: up
	mkdir -p mlflow-store runs-local/model runs-local/model-challenger
	docker build -q -t ch02-scorecard:latest scorecard/
	docker build -q -t ch02-challenger:latest challenger/
	docker run --rm $(MLFLOW) $(INPUT) -v $(PWD)/runs-local/model:/opt/ml/model ch02-scorecard:latest train
	docker run --rm $(MLFLOW) $(INPUT) -v $(PWD)/runs-local/model-challenger:/opt/ml/model ch02-challenger:latest train

# browse the runs on demand (no server runs otherwise)
mlflow-ui:
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 uvx mlflow ui \
		--backend-store-uri sqlite:///mlflow-store/mlruns.db --port 5000

# serve the scorecard locally (same image, same /ping,/invocations as the endpoint)
serve:
	docker rm -f ch02-serve 2>/dev/null || true
	docker run -d --name ch02-serve -p 8091:8080 -v $(PWD)/runs-local/model:/opt/ml/model:ro ch02-scorecard:latest serve

score:
	curl -s -X POST http://localhost:8091/invocations -H 'Content-Type: application/json' --data-binary @fixtures/sample.json; echo

# batch-score a file through a local serving container (parity with the endpoint)
batch:
	docker rm -f ch02-batch 2>/dev/null || true
	docker run -d --name ch02-batch -p 8094:8080 -v $(PWD)/runs-local/model-challenger:/opt/ml/model:ro ch02-challenger:latest serve
	sleep 4
	uv run serving/batch.py --url http://localhost:8094 --input data/split/test.csv --output /tmp/ch02-scored.csv
	docker rm -f ch02-batch 2>/dev/null || true

# Local AMT: Syne Tune runs the same search as managed AMT on your laptop — each
# trial a local subprocess, a Bayesian (TPE) scheduler, no instance and no Docker.
# Trials log to the serverless MLflow App, so this needs REAL AWS credentials: env -u
# strips the Makefile's S3Proxy placeholders (override with make amt APP_ARN=...).
APP_ARN ?= $(shell env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY aws sagemaker list-mlflow-apps --query "Summaries[?Name=='ch02-mlflow'].Arn | [0]" --output text)

amt:
	@test -n "$(APP_ARN)" && test "$(APP_ARN)" != "None" || \
		{ echo "no MLflow App named ch02-mlflow on this account: make mlflow-app (aws/) or pass APP_ARN=..."; exit 1; }
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
		MLFLOW_TRACKING_URI="$(APP_ARN)" uv run tuning/amt.py

# the local-tuning suite: trial contract, local/cloud search-space parity, and a
# two-trial TPE mini search — all offline, no Docker, no AWS
test:
	uv run --python 3.12 --with pytest --with syne-tune --with xgboost \
		--with scikit-learn --with pandas --with numpy pytest tests/ -q

# assemble + build the Lambda container (model baked in), then run it via the RIE
lambda-build:
	cp challenger/challenger_model.py serving/lambda/challenger_model.py
	mkdir -p serving/lambda/model && cp runs-local/model-challenger/challenger.ubj runs-local/model-challenger/feature_spec.json serving/lambda/model/
	docker build -q -t ch02-lambda:latest serving/lambda/

lambda-local: lambda-build
	docker rm -f ch02-lambda-rie 2>/dev/null || true
	docker run -d --name ch02-lambda-rie -p 9010:8080 ch02-lambda:latest
	sleep 3
	curl -s -XPOST http://localhost:9010/2015-03-31/functions/function/invocations --data-binary @fixtures/sample.json; echo
	docker rm -f ch02-lambda-rie 2>/dev/null || true

# assemble + build the Fargate image, then run it as a local ECS task (compose)
fargate-build: train-challenger
	mkdir -p serving/fargate/model && cp runs-local/model-challenger/challenger.ubj runs-local/model-challenger/feature_spec.json serving/fargate/model/
	docker build -q --build-arg BASE_IMAGE=ch02-challenger:latest -t ch02-fargate:latest serving/fargate/

fargate-local: fargate-build
	cd serving/fargate && docker compose up -d && sleep 5 && \
		curl -s -XPOST http://localhost:8093/invocations -H 'Content-Type: application/json' --data-binary @../../fixtures/sample.json; echo; \
		docker compose down

# the pipeline parity check: download the local-mode and cloud-mode score
# exports (pipeline/README.md runs the pipeline twice, once per prefix) and
# compare them byte for byte. Real AWS credentials, hence env -u.
LOCAL_SCORES ?= ch02/pipeline-scores-local
AWS_SCORES ?= ch02/pipeline-scores-aws
pipeline-parity:
	@env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY sh -c ' \
		aws s3 cp s3://$(SCORE_BUCKET)/$(LOCAL_SCORES)/scores.csv /tmp/ch02-scores-local.csv && \
		aws s3 cp s3://$(SCORE_BUCKET)/$(AWS_SCORES)/scores.csv /tmp/ch02-scores-aws.csv && \
		shasum -a 256 /tmp/ch02-scores-local.csv /tmp/ch02-scores-aws.csv && \
		cmp /tmp/ch02-scores-local.csv /tmp/ch02-scores-aws.csv && \
		echo "parity: scores.csv byte-identical"'

down:
	docker compose down
	docker rm -f ch02-serve ch02-batch ch02-lambda-rie 2>/dev/null || true

lint:
	uvx ruff check .