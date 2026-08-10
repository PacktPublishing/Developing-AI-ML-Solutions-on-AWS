# Monitoring models in production

What keeps a bad model from making bad decisions at scale: online drift
monitoring of the scores and the features that drive them, drift in the
scorecard's own feature contributions, and CloudWatch alerting on top.

## What it builds

- Online **Population Stability Index (PSI)** monitoring of scores and features.
  Features are binned at the model's own CatBoost split borders (the ML approach
  to PSI); the score, which has no model borders, is binned into reference
  quantiles. The fitted detector serialises to a JSON artifact and logs to
  **MLflow** as a pyfunc model, so the drift baseline carries the same lineage
  as the model it watches.
- **SHAP attribution drift**: rank features by mean absolute SHAP contribution
  and report the NDCG of the current ranking against the reference one,
  mirroring SageMaker Clarify's
  [feature-attribution drift](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-feature-attribution-drift.html).
- The SageMaker-shaped artifacts: Clarify's `analysis.json` and Model Monitor's
  `constraint_violations.json`, from one monitoring entrypoint
  (`src/run_monitor.py`) that runs locally and as the Processing step alike.
- **CloudWatch alerting** that fires before bad decisions accumulate: locally
  through a from-source shim, on AWS through the stack's alarms.

SageMaker Model Monitor and Clarify are closed to new accounts; AWS's
documented replacement is to run the same standardized metrics and SHAP
yourself as a Processing job. That is exactly the AWS round here: a SageMaker
Pipeline, Batch Transform (score the current batch) into Processing (the drift
monitor), reusing the book's SageMaker spine.

## How it runs

Locally, in the SageMaker local-mode contract first:

```
make up        # the local S3 (S3Proxy) the artifacts land in -- optional, for the S3 step
make data      # generate the reference and current batches
make train     # the /opt/ml training image, run as a container
make monitor   # the drift report (PSI + attribution NDCG)
make run       # the monitoring job: artifacts, metrics, alarms (MONITOR_LOCAL=1)
make down      # stop the local S3
```

`make run` writes the artifacts to `outputs/monitoring/` and, when the local S3 store is
up (`make up`), uploads them to `s3://ch11-monitoring/` -- the stand-in for SageMaker
uploading `/opt/ml/processing/output` to S3 on the managed run.

On AWS (`aws/`): `make iam`, `make deploy` (ECR repository + the CloudWatch
alarms), `make image`, `make pipeline` (Batch Transform, then the Processing
step running the same `run_monitor.py` with the `/opt/ml/processing` paths as
arguments).

## Layout

- `src/`: `model.py` (the CatBoost scorecard), `psi.py` (PSIDetector + the
  MLflow logging), `attribution.py` (SHAP ranking + NDCG), `monitor.py` (the
  monitors and the violations report), `cloudwatch.py` (the alerting client:
  real boto3 or the local shim), `run_monitor.py` (the one monitoring
  entrypoint), `train.py`, `serve.py`
- `Dockerfile` + `entrypoint.sh`: the train/serve/monitor image; the same image
  runs locally and as the managed training, transform, and Processing jobs
- `aws/`: `template.yaml` (ECR + alarms), `pipeline.py` (Transform into
  Processing), `iam/deploy.json` (the ch11-user deploy policy)
- `etl/make_batches.py`: the synthetic reference and drifted current batches
- `local/monitoring-stack.yml`: S3Proxy, the local S3 the artifacts upload to
- `notebooks/monitoring.ipynb`: the narrative over the tested `src/`
- `tests/`, `diagrams/`

## AWS services and local stand-ins

- **AWS services:** Amazon SageMaker (Pipelines, Batch Transform, Processing),
  Amazon CloudWatch, Amazon S3, Amazon ECR
- **Local stand-ins:** the same SageMaker container contract run with plain
  `docker run`; a from-source CloudWatch shim; **S3Proxy** for the artifact
  bucket the Processing step writes to; SHAP + CatBoost run directly
