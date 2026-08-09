.PHONY: data train monitor run lint clean

export AWS_ACCESS_KEY_ID ?= local
export AWS_SECRET_ACCESS_KEY ?= localsecret
export AWS_DEFAULT_REGION ?= us-east-1

IMAGE := ch11-scorecard:latest
# The /opt/ml training channels: the reference batch is the channel the scorecard trains on.
INPUT := -v $(PWD)/runs-local/input:/opt/ml/input:ro -v $(PWD)/runs-local/output:/opt/ml/output

# synthesize the reference and current (drifted) batches, then lay out the training channel
data:
	uv run etl/make_batches.py
	mkdir -p runs-local/input/data/reference runs-local/input/config runs-local/output
	cp data/generated/reference.csv runs-local/input/data/reference/
	printf '{"iterations":"300","depth":"5","learning_rate":"0.1"}' > runs-local/input/config/hyperparameters.json

# train the scorecard in SageMaker local mode: the /opt/ml contract image, run as a container
train:
	docker build -q -t $(IMAGE) .
	rm -rf runs-local/model && mkdir -p runs-local/model
	docker run --rm $(INPUT) -v $(PWD)/runs-local/model:/opt/ml/model $(IMAGE) train

# the drift monitors: PSI on scores and features, SHAP attribution NDCG
monitor:
	PYTHONPATH=src uv run src/monitor.py --model runs-local/model/scorecard.cbm

# the monitoring job: SageMaker-shaped artifacts plus CloudWatch metrics and alarms (local shim)
run:
	MONITOR_LOCAL=1 PYTHONPATH=src uv run src/run_monitor.py --model runs-local/model/scorecard.cbm

lint:
	uvx ruff check .

clean:
	rm -rf runs-local outputs data/generated catboost_info