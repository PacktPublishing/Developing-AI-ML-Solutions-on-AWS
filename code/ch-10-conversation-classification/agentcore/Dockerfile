# AgentCore Runtime requires an ARM64 image serving port 8080. Building on an
# Apple silicon laptop is already ARM64, so the image that runs here is the
# image that runs in the cloud.
FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Which torch to install, and it decides whether the image can be deployed at
# all. The default aarch64 wheel bundles CUDA for ARM server GPUs: 3.3 GB of
# nvidia plus 916 MB of triton, which builds a 9.39 GB image against AgentCore
# Runtime's 2 GB limit. The CPU index drops both and builds 1.47 GB.
#   cpu (default): https://download.pytorch.org/whl/cpu
#   gpu:           https://pypi.org/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# Dependencies first, so editing the agent does not reinstall torch.
COPY requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache --index-url "$TORCH_INDEX" torch \
    && uv pip install --system --no-cache -r requirements.txt

COPY app.py ./app.py

# The weights are not baked in. The checkpoint plus its Qwen3.5-2B-Base torso is
# about 4.8 GB, and neither artifact AgentCore accepts can carry that: a
# container image stops at 2 GB and an S3 source package at 750 MB unpacked. The
# model is fetched into a mounted volume at session start instead.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models \
    PORT=8080

EXPOSE 8080

CMD ["python", "app.py"]
