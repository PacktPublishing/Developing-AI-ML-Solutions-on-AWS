# The scoring Lambda that loads from the MLflow registry at runtime (no model baked
# into the image). mlflow-skinny + the sagemaker-mlflow plugin reach the registry; on a
# SageMaker MLflow App that is all you need. sqlalchemy + alembic are only for the local
# sqlite store used to test this offline; catboost/pandas load and score the native flavor.
# Deps install with uv (fast, reproducible). challenger_model.py is copied into the build
# context by the make targets. Built in-region by CodeBuild (amd64); the SAM function
# references the pushed image so a ~2 GB image never leaves your laptop's uplink.
FROM public.ecr.aws/lambda/python:3.12

RUN pip install --no-cache-dir uv && uv pip install --system --no-cache \
    "mlflow-skinny>=3.10,<4" \
    "sagemaker-mlflow>=0.1" \
    "sqlalchemy>=2.0" \
    "alembic>=1.13" \
    "catboost>=1.2" \
    "pandas>=2.0" \
    "numpy>=1.26" \
    "boto3>=1.34"

# Lambda's only writable path is /tmp; point temp files (artifact downloads) there.
ENV TMPDIR=/tmp

COPY challenger_model.py handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
