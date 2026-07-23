# Champion/challenger pipeline (one script, local and cloud)

The chapter's CEO question — "why this model, did you try others?" — as a SageMaker
Pipeline: train the logistic scorecard and the monotone XGBoost challenger, select
the winner by AUC, and export the winner's scores to S3.

`pipeline.py` runs the **same DAG** in local Docker and on AWS. Only the session and
instance type change, driven by one environment variable:

| `PIPELINE_MODE` | Session | Instance | Compute |
|---|---|---|---|
| `local` (default) | `LocalPipelineSession` | `local` | Docker on the laptop |
| `aws` | `PipelineSession` | `ml.m5.large` | SageMaker Processing jobs |

## The DAG

Five traditional `ProcessingStep`s, data flowing between them by S3 property
reference (`step.properties.ProcessingOutputConfig.Outputs[...].S3Output.S3Uri`):

```
prepare ─┬─> train-scorecard ─┐
         └─> train-challenger ─┴─> select ─> export ─> s3://<bucket>/<prefix>/scores.csv
```

Each step is a `ScriptProcessor` running one script from `scripts/` on the shared
image: `prepare.py`, `train.py` (`--model scorecard|challenger`), `select.py`,
`export.py`. The **export is a genuine SageMaker Processing job** whose
`ProcessingOutput` is uploaded to S3 — not a Lambda step.

## Run it

```
docker build -t ch02-step:local pipeline/step_image/

# local — no instance quota, containers run on your machine
PIPELINE_MODE=local STEP_IMAGE=ch02-step:local SCORE_BUCKET=<bucket> \
SAGEMAKER_ROLE_ARN=arn:aws:iam::<acct>:role/<SageMakerExecutionRole> \
  uv run pipeline/pipeline.py

# cloud — same script, ECR image, real SageMaker jobs
PIPELINE_MODE=aws INSTANCE_TYPE=ml.m5.large \
STEP_IMAGE=<acct>.dkr.ecr.us-east-1.amazonaws.com/ch02-pipeline-step:latest \
SCORE_BUCKET=<bucket> \
SAGEMAKER_ROLE_ARN=arn:aws:iam::<acct>:role/<SageMakerExecutionRole> \
  uv run pipeline/pipeline.py
```

Both runs select the same winner (the challenger, AUC 0.7503 vs the scorecard's
0.7485) and export a byte-identical 800-row `scores.csv`.

## Notes that cost time to learn (SDK v3)

- **Traditional steps, not `@step`.** `@step` serializes its arguments eagerly, so a
  `@step` result cannot feed a `ProcessingStep` (`DelayedReturn is not JSON
  serializable`). Traditional steps flow data through `step.properties`, which
  serialize lazily as pipeline variables — the pattern in AWS's local-mode example.
- **v3 splits the SDK into packages, and `pip install sagemaker` is only core +
  train.** The pipeline pieces live in `sagemaker-mlops`, and `sagemaker-mlops`
  reaches into `sagemaker.serve` at import time without depending on it, so the
  script needs `sagemaker-mlops` AND `sagemaker-serve` declared explicitly.
- **v3 ships two `LocalPipelineSession`s, each with half the job.** The one in
  `sagemaker.core.workflow.pipeline_context` inherits `PipelineSession`, so
  `Processor.run` defers into step arguments — but it has no pipeline methods. The
  one in `sagemaker.mlops.local` implements `create_pipeline` and
  `start_pipeline_execution` — but does not inherit `PipelineSession`, so every
  step executes immediately at definition time. `pipeline.py` subclasses both,
  which is the whole local-mode fix.
- **Local registration goes through the session, not `upsert`.**
  `Pipeline.upsert`/`start` call `sagemaker_client.create_pipeline`, which local
  mode does not implement; the local session carries those methods itself. On AWS,
  `upsert` + `start` work as documented.
- **Explicit S3 URIs on every output.** v3's `ProcessingOutput` nests an
  `s3_output` with a required `s3_uri` (v2 auto-generated one per job), so
  step-to-step I/O gets a deterministic prefix (`IO_PREFIX`). Local mode still
  stages I/O in S3: credentials and a bucket are required even though the
  containers run locally — only the compute avoids AWS.

## `step_image/`

`Dockerfile` for the step runtime (sagemaker + scikit-learn + xgboost + pandas +
numpy + joblib), built with **uv** (pip stalled 10+ minutes resolving sagemaker's
dependency tree; uv finished in ~3 minutes). Build native arm64 for local mode; the
`ch02-pipeline-step` ECR image is amd64 for the cloud run.
