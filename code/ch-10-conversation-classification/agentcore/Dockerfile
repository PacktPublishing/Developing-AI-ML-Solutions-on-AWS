# The platform follows the compute type, and they disagree. microVMs take ARM64
# only, which is what an Apple silicon laptop builds natively. Every GPU family
# AgentCore supports is x86_64, so the deployed image is linux/amd64 and
# CodeBuild builds it in the account rather than cross-building on a laptop.
ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Only the agent and the runtime SDK. Measured on arm64, an image carrying torch
# as well is 9.39 GB with the CUDA wheel and 1.47 GB with the CPU one, against a
# 2 GB limit that is not adjustable, so neither can ship. uv stays on PATH from
# the base image because the prepare call uses it to install torch and the model
# onto the volume at session start.
COPY requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

COPY app.py ./app.py

# MODEL_VOLUME is where the capacity provider mounts its EBS volume: torch, the
# model, and the Hugging Face cache all live there and survive a session stop.
ENV PYTHONUNBUFFERED=1 \
    MODEL_VOLUME=/mnt/models \
    HF_HOME=/mnt/models/hf \
    PORT=8080

EXPOSE 8080

CMD ["python", "app.py"]
