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
# The CPU variant of the same image runs this on a laptop, which is what lets
# SageMaker local mode exercise the real startup contract before a deploy.
ARG BASE_IMAGE=public.ecr.aws/sagemaker/sagemaker-distribution:4.7.0-gpu
FROM ${BASE_IMAGE}

ARG DECIDER_VERSION=0.3.0
ARG CHECKPOINT=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610

# The image runs as a non-root user; installing into its environment needs root.
USER root

# No --index-url and no torch here on purpose. Anything that reinstalls torch
# puts back the mismatch this base image exists to avoid.
RUN pip install --no-cache-dir "strands-decider==${DECIDER_VERSION}" \
    && python -c "import torch; print('torch', torch.__version__)"

# Bake the weights in. SageMaker gives /invocations 60 seconds, so a 4.5 GB
# download belongs in the build rather than in the first request.
ENV HF_HOME=/opt/models
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${CHECKPOINT}'); snapshot_download('Qwen/Qwen3.5-2B-Base')" \
    && chmod -R a+rX /opt/models

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
ENTRYPOINT ["./entrypoint.sh"]
