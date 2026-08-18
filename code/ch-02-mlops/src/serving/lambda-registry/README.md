# Registry-loading Lambda

Serves the challenger by loading it from the MLflow registry at cold start, instead of
baking the model into the image (the counterpart to `../lambda/`). `mlflow-skinny` plus
the `sagemaker-mlflow` plugin resolve the registered version and download its run
artifacts to `/tmp`; `catboost` loads the native flavor and scores.

## Local (sqlite registry + S3Proxy)

    make up                      # local stack (S3Proxy)
    make mlflow-train SM_OFFLINE=1   # register credit-challenger in the local sqlite registry
    make lambda-registry-local   # build + run under the Lambda RIE, invoke fixtures/sample.json

The local sqlite tracking store additionally needs `sqlalchemy` + `alembic` in the image
(already in the Dockerfile); the AWS MLflow App is REST, so those are unused there.

## AWS (SageMaker MLflow App) -- verified

Verified end to end on 2026-08-17: the model came from the App registry, not the image.

    make -C aws mlflow-app                 # the MLflow App (if not already up) + register credit-challenger
    # image: build in-region with CodeBuild (2 GB with catboost; do not push from a laptop)
    make -C aws image-bootstrap REPO=ch02-lambda-registry
    make -C aws image REPO=ch02-lambda-registry SRCDIR=serving/lambda-registry
    make -C aws lambda-registry-deploy     # role + Image function against the App ARN
    make -C aws lambda-registry-invoke     # -> {"pd": [0.9958..., 0.0001...]}
    make -C aws lambda-registry-teardown   # + make -C aws image-teardown REPO=ch02-lambda-registry

Test the packaged image with `sam local invoke`:

    cd aws/lambda-registry
    sam local invoke RegistryFunction \
      --parameter-overrides "ImageUri=ch02-lambda-registry:latest MlflowAppArn=<App ARN>" \
      -e ../../fixtures/sample.json      # -> loaded credit-challenger v1 ... {"pd": [...]}

See `diagrams/fig_lambda_registry` for the flow.

Execution-role permissions the function needs:

- `sagemaker-mlflow:*` AND `sagemaker:*` to reach the MLflow App. `sagemaker-mlflow:*`
  alone returns `403 Request is not authorized` from the App's REST endpoint; the
  authorizer also checks a `sagemaker:` action.
- `s3:GetObject` / `s3:ListBucket` on the App's artifact bucket
  (`ch02-sagemaker-mlflow-<account>-<region>`), where the run artifacts live.

## The trade

This route consults the registry at wake time, so a cold start pays the ~2 GB image pull
plus the version-resolve and artifact download. The shipped `../lambda/` bakes the model in
(registry at build time) and is the lighter path; use this one when you want the running
model to follow the registry without a rebuild.
