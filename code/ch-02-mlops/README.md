# MLOps on AWS: training, tracking, and the serving surface

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
endpoint. The chapter uses **SageMaker Python SDK v3** throughout — cloud and
local (`sagemaker.train.ModelTrainer`, `sagemaker.serve.ModelBuilder`,
`sagemaker.train.tuner.HyperparameterTuner`, and the pipeline in `pipeline/`);
where v3's local mode has gaps, the chapter fills them in place
(`pipeline/pipeline.py`) rather than falling back to v2. Local hyperparameter
search runs on **Syne Tune** (the AMT team's open-source tuner, `tuning/amt.py`);
the managed search is SageMaker AMT (`aws/jobs/amt.py`).

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
  SageMaker training job on AWS (`aws/train_job.py`, v3 `ModelTrainer`). The
  image is built in the cloud by CodeBuild -> ECR, so an Apple-silicon laptop
  never cross-builds under emulation.
- **Serving**: the same container is a local `docker run`, a SageMaker serverless
  endpoint, a Lambda, and an ECS Fargate task — and returns the same score in
  every one.

The `aws/` folder reproduces each step on real AWS.

## The serving surface (all run on real AWS, parity-checked)

- **Real-time**: a serverless SageMaker endpoint from the registry. `make serve`
  is the local equivalent (same image).
- **Batch**: stream a file through the endpoint (`serving/batch.py`). Note the
  ~6 MB payload and ~60 s timeout per invoke, so batches chunk; a true Batch
  Transform job is the alternative for very large files.
- **Lambda + container**: the scoring model as a Lambda container image, tested
  locally with the open-source Runtime Interface Emulator.
- **ECS Fargate**: the same serving image as a long-lived task; local emulation
  via `amazon-ecs-local-container-endpoints` and docker compose.

## Layout

- `docker-compose.yml`: the local cloud (S3Proxy)
- `data/generate_applications.py`: synthetic credit data + the shared monotone spec
- `scorecard/`: the incumbent WOE scorecard container (fastwoe + LogisticRegression)
- `challenger/`: the monotone XGBoost challenger container
- `tuning/amt.py`: local HPO with Syne Tune (Bayesian TPE, no instance), tracked in MLflow
- `serving/batch.py`, `serving/lambda/`, `serving/fargate/`: the serving surfaces
- `aws/`: reproduce on real AWS (SDK v3), with IAM notes
- `Makefile`: the targets above; `make lint` runs ruff
