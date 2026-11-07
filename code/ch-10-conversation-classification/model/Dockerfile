# The decision model behind a SageMaker endpoint. x86_64, because every GPU
# instance family is, and SageMaker has no image size limit, so CUDA torch and
# the weights ship in the image rather than being installed at session start.
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG DECIDER_VERSION=0.3.0
ARG CHECKPOINT=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610

# No bespoke inference code. strands-decider's own server already answers
# POST /invocations and GET /ping, which is exactly the contract SageMaker
# requires of a container, and it warms the model before reporting ready.
RUN uv pip install --system --no-cache "strands-decider[cuda]==${DECIDER_VERSION}"

# Bake the weights in. SageMaker gives /invocations 60 seconds, so the 4.5 GB
# download belongs in the build rather than in the first request.
ENV HF_HOME=/opt/models
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${CHECKPOINT}'); snapshot_download('Qwen/Qwen3.5-2B-Base')"

ENV PYTHONUNBUFFERED=1 \
    CHECKPOINT=${CHECKPOINT}

EXPOSE 8080

# SageMaker starts a container with `docker run <image> serve`, and that argument
# replaces CMD. A command named serve has to exist on PATH, or the container
# exits with CannotStartContainerError before anything is logged. 0.0.0.0 rather
# than the default 127.0.0.1, because SageMaker reaches it from outside.
RUN printf '%s\n' '#!/bin/sh' \
    'exec strands-decider serve "$CHECKPOINT" --host 0.0.0.0 --port 8080 --device cuda' \
    > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

CMD ["serve"]
