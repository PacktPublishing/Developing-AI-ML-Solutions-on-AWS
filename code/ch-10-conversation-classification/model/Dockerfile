# The decision model behind a SageMaker endpoint, on SageMaker Distribution.
#
# The base image matters more than anything else here. A plain image with no
# torch means pip installs the default wheel, which is built for a newer CUDA
# than SageMaker's GPU hosts carry, and the container dies at startup with
# "NVIDIA driver is too old" before it can answer /ping. SageMaker Distribution
# ships pytorch 2.8.0 built against CUDA 12.9, matched by AWS to its own
# hardware, so the right move is to use the torch that is already there rather
# than pin one. strands-decider wants torch>=2.7 and 2.8.0 satisfies it, so pip
# leaves it alone.
#
# The PyTorch inference DLC would be the usual choice for an endpoint, but it
# stops at torch 2.6.0, below what this model needs.
# SageMaker Distribution publishes amd64 only, so on an Apple silicon laptop it
# can only run emulated, and emulating a 7 GB base plus 4.5 GB of weights is a
# build measured in hours. The local-mode check does not need this base: it
# checks how the container starts and what serve.py does, not which image the
# endpoint runs. So the base is an argument, and `make sm-local` passes a
# multi-arch slim one that builds natively.
ARG BASE_IMAGE=public.ecr.aws/sagemaker/sagemaker-distribution:4.7.0-gpu
FROM ${BASE_IMAGE}

ARG DECIDER_VERSION=0.3.0
ARG CHECKPOINT=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
# Empty means the base already carries torch, which is the whole reason the
# endpoint uses SageMaker Distribution. A base without one needs an index, and
# it must be the CPU index: the default wheel bundles CUDA nobody here can use.
ARG TORCH_INDEX=
# The weights are baked in for the endpoint, where SageMaker gives /invocations
# 60 seconds. A local contract check does not need 4.5 GB to prove the container
# starts the right server.
ARG BAKE_WEIGHTS=1

# The image runs as a non-root user; installing into its environment needs root.
USER root

# No --index-url and no torch here on purpose. Anything that reinstalls torch
# puts back the mismatch this base image exists to avoid.
RUN if [ -n "$TORCH_INDEX" ]; then \
        pip install --no-cache-dir --index-url "$TORCH_INDEX" torch; \
    fi \
    && pip install --no-cache-dir "strands-decider==${DECIDER_VERSION}" \
    && python -c "import torch; print('torch', torch.__version__)"

# Bake the weights in. SageMaker gives /invocations 60 seconds, so a 4.5 GB
# download belongs in the build rather than in the first request.
#
# A decider is a LoRA adapter over a base model, and both have to be here.
# Which base it needs is the checkpoint's to say, so read it out of
# strands_decider_config.json rather than hardcode one.
ENV HF_HOME=/opt/models
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        python -c "import json; from huggingface_hub import snapshot_download; \
path = snapshot_download('${CHECKPOINT}'); \
base = json.load(open(path + '/strands_decider_config.json'))['base_model']; \
print('base model:', base); \
snapshot_download(base)" \
        && chmod -R a+rX /opt/models; \
    fi

WORKDIR /opt/program
COPY serve.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# cuda on an endpoint, cpu when local mode runs this where there is none.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program \
    CHECKPOINT=${CHECKPOINT} \
    DEVICE=cuda

EXPOSE 8080

# SageMaker starts the image as `... serve`, and that argument would replace a
# CMD. An ENTRYPOINT takes it as an argument instead, which is why every serving
# image in this book is started this way rather than through a command called
# serve on PATH.
#
# Overriding the base image's own ENTRYPOINT is the point here, not a detail.
# SageMaker Distribution ships an inference server of its own, at
# /etc/sagemaker-inference-server, which expects a handler at
# /opt/ml/model/inference.py and ignores a CMD entirely. Either write that
# handler and let its server call you, or take the entrypoint and run your own.
# This runs its own, because strands-decider already serves the System One
# contract and an inference.py would mean reimplementing the schema.
ENTRYPOINT ["./entrypoint.sh"]
