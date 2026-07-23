# Cloud reproduction

Everything in the chapter runs locally first; this folder reproduces each step on
real AWS with the SageMaker Python SDK v3. Replace the account-specific values (account id, bucket, role ARNs, ECR image
URIs, the MLflow App ARN) with your own before running.

## The image build: local vs CI

Locally, the container image is built by Docker Desktop and is right there on your
machine — local mode needs no registry. For the cloud, the image is built by a CI
pipeline and pushed to a registry:

- Developer -> Git -> **CodeBuild** -> **Amazon ECR** (ECR is the AWS equivalent of
  Docker Hub). `buildspec.yml` is the build; `codebuild_*` in the chapter diagrams
  is the shape.
- A CI trigger belongs in front of CodeBuild. `github-actions.yml` is an example
  GitHub Actions workflow that starts the CodeBuild project on push.

On an Apple-silicon laptop this also avoids cross-building the amd64 image under
emulation (which is far too slow) — CodeBuild builds natively.

## Layout

The `Makefile` is the entry point — it derives the account, bucket, execution
role, and MLflow App ARN, so each step is one target instead of a long env-var
line. Override any value on the command line (e.g. `make train ROLE=...`).

```
aws/
  Makefile          one target per step (make mlflow-app / train / amt / deploy-byoc ...)
  mlflow_app.sh     create / describe / delete the serverless MLflow App
  image/            build the container in the cloud
    buildspec.yml       the CodeBuild build
    github-actions.yml  example CI that triggers it on push
  jobs/             SageMaker jobs (need training-instance quota)
    train_job.py        training job (v3 ModelTrainer)
    amt.py              Automatic Model Tuning (v3 HyperparameterTuner)
  deploy/           deploy a registered model, two ways
    deploy_byoc_from_registry.py   on the BYOC image (pin env in the image)
    deploy_modelbuilder.py         via ModelBuilder DLC (pin env in the model)
    mb-requirements.txt            serving deps for the ModelBuilder path
```

The one-time setup the `Makefile` assumes (ECR repos, CodeBuild projects, IAM
roles) is created interactively the first time; the serving surfaces `batch`,
`lambda`, and `fargate` have their own local and cloud paths under `serving/`.

## Cost and quota notes

- The whole chapter, including every real-AWS parity check, cost effectively
  nothing on a free-tier account (month-to-date ~$0 at the time of the runs).
- A fresh account's **training and transform instance quota is 0** — SageMaker
  *training jobs*, *tuning jobs*, and *batch-transform jobs* will not run until you
  request an increase (Service Quotas, e.g. "ml.m5.large for training job usage").
- **Serverless serving does not need instance quota**: serverless inference
  endpoints (max 3072 MB memory on the default quota) and Lambda both run as-is,
  which is why the chapter serves models the serverless / BYOC way.
- A **Fargate** service bills continuously (no scale-to-zero); stop tasks when done.
- Tear down: delete endpoints and endpoint-configs and models when finished;
  `aws lambda delete-function`; stop Fargate tasks / delete the cluster; the
  serverless MLflow App is free at rest but can be deleted with `mlflow_app.sh delete`.

## IAM

The runs used an admin-ish user plus these managed policies on the account's group,
and a few purpose-built roles. Scope every resource to your account and region in
production.

| Area | Permissions (managed policy or actions) | Used for |
|---|---|---|
| Images | `AmazonEC2ContainerRegistryFullAccess` (`ecr:*`) | BYOC images to ECR |
| CI build | `AWSCodeBuildAdminAccess` (`codebuild:*`) + role `ch02-codebuild-role` (ECR power-user, logs, S3 read) | build images in the cloud |
| SageMaker | `AmazonSageMakerFullAccess` (`sagemaker:*`, incl. `InvokeEndpoint`, MLflow Apps, model packages) + a SageMaker execution role | train, register, serve |
| Lambda | `AWSLambda_FullAccess` + role `ch02-lambda-role` (basic execution) | container Lambda |
| Fargate | `AmazonECS_FullAccess` (+ `ec2:*` networking) + role `ch02-ecs-exec-role` (ECS task execution) | ECS Fargate task |
| Storage / logs | `AmazonS3FullAccess`, CloudWatch Logs | artifacts, model data, container logs |
| Roles | `iam:PassRole` | pass the execution/task roles to the services |
