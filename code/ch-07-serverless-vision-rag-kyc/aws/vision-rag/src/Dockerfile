# facenet face-embedding Lambda (CPU-only). One image, two handlers — the
# template's ImageConfig.Command picks embeddings_handler or search_handler per
# function. Built with uv, following docs.astral.sh/uv/guides/integration/docker.
FROM public.ecr.aws/lambda/python:3.12

# uv binary, pinned for reproducible builds (official distroless image).
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_PYTHON_DOWNLOADS=0 \
    TORCH_HOME=/opt/torch

# CPU-only torch/torchvision from the PyTorch CPU index (Lambda has no GPU), then
# the embedder + Postgres deps. boto3 already ships in the Lambda base image.
# The cache mount keeps wheels between builds (uv guide).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install facenet-pytorch "numpy<2" pillow "psycopg[binary]" captum

# Bake the VGGFace2 weights (public facenet-pytorch release) into the image so
# cold starts need no internet access and no private bucket. chmod so the
# non-root Lambda runtime user can read the root-baked cache.
RUN python -c "from facenet_pytorch import InceptionResnetV1, MTCNN; \
    InceptionResnetV1(pretrained='vggface2').eval(); MTCNN()" \
    && chmod -R a+rX /opt/torch

# Shared embedder + explainer + the three handlers.
COPY face_embedder.py face_explainer.py embeddings_handler.py \
     search_handler.py compare_handler.py ${LAMBDA_TASK_ROOT}/

# Default command; overridden per function by ImageConfig.Command in template.yaml.
CMD ["search_handler.lambda_handler"]
