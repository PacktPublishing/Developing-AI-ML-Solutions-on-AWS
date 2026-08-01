# Runtime image for the pipeline's processing steps. Built with uv and
# natively for the host architecture: SageMaker local mode runs this
# container on your machine; the cloud run uses the same Dockerfile built
# for amd64.
#
# Build:  docker build -t ch05-step:local pipeline/step_image/
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1

RUN uv pip install --system \
    "catboost>=1.2" \
    "pandas>=2.0" \
    "numpy>=1.26" \
    "psycopg2-binary>=2.9"

# the SageMaker training entrypoint: `docker run <image> train`
COPY train /usr/local/bin/train
RUN chmod +x /usr/local/bin/train
