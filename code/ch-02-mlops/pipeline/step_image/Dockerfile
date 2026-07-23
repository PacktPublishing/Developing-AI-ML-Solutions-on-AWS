# Runtime image for the @step functions in local_pipeline.py. A v3 @step compiles
# to a SageMaker remote-function job, so its container needs the sagemaker runtime
# (to deserialize and invoke the function) plus this chapter's training libraries.
#
# Built with uv, not pip: resolving sagemaker's dependency tree with pip stalled
# for 10+ minutes; the same install with uv finishes in ~3 minutes. Built natively
# for the host architecture (arm64 on Apple silicon) because SageMaker local mode
# runs the container on your machine, not on an amd64 instance -- no emulation.
#
# Build:  docker build -t ch02-step:local pipeline/step_image/
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1

RUN uv pip install --system \
    "sagemaker>=3,<4" \
    "scikit-learn>=1.3" \
    "xgboost>=2.0" \
    "pandas>=2.0" \
    "numpy>=1.26" \
    "joblib>=1.3"
