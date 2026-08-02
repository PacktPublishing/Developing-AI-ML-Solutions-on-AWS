# MLOps on AWS: training, tracking, and the serving layer

The running example for the MLOps chapter. A mid-sized bank is moving its credit
scoring onto SageMaker. Its established model is a Weight-of-Evidence
logistic-regression scorecard; a data scientist tests a monotone XGBoost
challenger against it, tunes it, and then has to choose how to serve the winner.
Every model here obeys the same business rule — risk moves monotonically with
certain features — the scorecard by construction, the challenger through
`monotone_constraints`.

The local cloud is real engines in Docker, no mocks and no LocalStack:

- S3Proxy serving the S3 API (MLflow artifacts, training input/output, model
  tarballs, batch data). On AWS this is Amazon S3; the code does not change.
- MLflow with no tracking server. SageMaker AI's serverless MLflow ("MLflow
  Apps") has no server to run; the local mirror is the same shape — the MLflow
  client writes runs and the model registry to a sqlite file and artifacts
  straight to S3Proxy. The one thing that moves between worlds is
  `MLFLOW_TRACKING_URI`: a sqlite path locally, an MLflow App ARN on AWS.

Models train and serve in **custom containers** (bring-your-own-container): one
image both trains (the SageMaker `/opt/ml` contract) and serves (`/ping` and
`/invocations`), so the exact image that produced a model is the one behind the
endpoint. The chapter uses **SageMaker Python SDK v3** for the managed work on
AWS: `sagemaker.train.ModelTrainer` runs the training job,
`sagemaker.train.tuner.HyperparameterTuner` runs Automatic Model Tuning, and
`sagemaker.serve.ModelBuilder` deploys the winner to an endpoint. The same container
runs locally two ways. Directly under Docker (`make train`, `make serve`,
`make batch` — plain `docker run`s), so the laptop exercises the exact image the
cloud runs. And through the SDK's **local mode**, the same deploy code the cloud
uses with the mode swapped: `make sm-local` serves it as a local SageMaker endpoint
(`Mode.LOCAL_CONTAINER`) and `make sm-batch` runs a local SageMaker Batch Transform
job on it (`make batch` is a hand loop over `/invocations`; `sm-batch` is the real
managed job, run locally). The pipeline (`src/pipeline/pipeline.py`) runs on a
`LocalPipelineSession` too. Local hyperparameter search runs on **Syne Tune** (the AMT team's
open-source tuner, `src/tuning/amt.py`); the managed search is SageMaker AMT
(`aws/jobs/amt.py`).

## Run it

```
make up               # start the local cloud (S3Proxy)
make seed             # synthesize the dataset and lay out the training channels
make train            # train the incumbent scorecard in its container
make train-challenger # train the monotone XGBoost challenger
make mlflow-train     # train both, logged to serverless MLflow (sqlite + S3Proxy)
make mlflow-ui        # browse the runs on demand (no server runs otherwise)
make amt              # local hyperparameter search on the challenger (Syne Tune), tracked in MLflow
make serve && make score   # serve the scorecard locally and score fixtures/sample.json
make batch            # batch-score a file through a local serving container
make lambda-local     # build the Lambda container and run it via the built-in RIE
make fargate-local    # build the Fargate image and run it as a local ECS task (compose)
make down             # stop everything
make lint             # ruff
```

## Local vs AWS

The code never changes; environment seams switch worlds.

- **Storage**: S3Proxy locally, Amazon S3 on AWS (same S3 API).
- **Experiment tracking / registry**: `MLFLOW_TRACKING_URI` is a sqlite path
  locally, a serverless MLflow App ARN on AWS (with the `sagemaker-mlflow` plugin).
- **Training**: `docker run ... train` locally; the same image submitted as a
  SageMaker training job on AWS (`aws/jobs/train_job.py`, v3 `ModelTrainer`). The
  image is built in the cloud by CodeBuild -> ECR, so an Apple-silicon laptop
  never cross-builds under emulation.
- **Serving**: the same container is a local `docker run`, a SageMaker serverless
  endpoint, a Lambda, and an ECS Fargate task — and returns the same score in
  every one.

The `aws/` folder reproduces each step on real AWS.

## The serving layer (all run on real AWS, parity-checked)

- **Real-time**: a serverless SageMaker endpoint from the registry. `make serve`
  is the local equivalent (same image).
- **Batch**: stream a file through the endpoint (`src/serving/batch.py`). Note the
  ~6 MB payload and ~60 s timeout per invoke, so batches chunk; a true Batch
  Transform job is the alternative for very large files.
- **Lambda + container**: the scoring model as a Lambda container image, tested
  locally with the open-source Runtime Interface Emulator.
- **ECS Fargate**: the same serving image as a long-lived task; local emulation
  via `amazon-ecs-local-container-endpoints` and docker compose.

## Layout

- `local/mlflow-stack.yml`: the local cloud (S3Proxy)
- `data/generate_applications.py`: synthetic credit data + the shared monotone spec
- `src/scorecard/`: the incumbent WOE scorecard container (fastwoe + LogisticRegression)
- `src/challenger/`: the monotone XGBoost challenger container
- `src/tuning/amt.py`: local HPO with Syne Tune (Bayesian TPE, no instance), tracked in MLflow
- `src/serving/batch.py`, `src/serving/lambda/`, `src/serving/fargate/`: the serving options
- `aws/`: reproduce on real AWS (SDK v3), with IAM notes
- `Makefile`: the targets above; `make lint` runs ruff
