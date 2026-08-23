# The KYC face service behind the SageMaker serving contract, on a GPU instance.
# Composed from source on a slim base rather than pulling a deep-learning container:
# the CUDA runtime arrives with the PyTorch wheels, and the instance supplies the driver.
FROM public.ecr.aws/docker/library/python:3.12-slim

# CUDA build of torch, pinned twice over. The version has to satisfy facenet-pytorch
# (torch<2.3), and the CUDA build has to be one the host driver supports: a cu121 image
# on this instance enumerates the device but fails to initialise it, which surfaces as
# device_count 1 alongside is_available False. cu118 is the conservative choice.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118
RUN pip install --no-cache-dir \
    facenet-pytorch "numpy<2" "scipy<1.14" pillow "psycopg[binary]" psycopg-pool captum boto3 \
    "sagemaker>=3,<4" "flask>=3.0" "gunicorn>=21.2"

# Bake the VGGFace2 weights in, so a scale-from-zero cold start needs no internet.
ENV TORCH_HOME=/opt/torch
RUN python -c "from facenet_pytorch import InceptionResnetV1, MTCNN; \
    InceptionResnetV1(pretrained='vggface2').eval(); MTCNN()" \
    && chmod -R a+rX /opt/torch

WORKDIR /opt/program
COPY serve.py entrypoint.sh face_embedder.py face_explainer.py face_spec.py kycstore.py ./
RUN chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/opt/program
ENTRYPOINT ["./entrypoint.sh"]
