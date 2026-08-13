# The fraud model's serving container: the image behind the SageMaker
# Serverless Inference endpoint, and the same image `make serve` runs locally.
# Composed from source on a slim base — no SageMaker deep-learning container
# to pull. The model artifact is not baked in: SageMaker mounts the model
# tarball at /opt/ml/model, and the local run mounts artifacts/ at the same
# path.
FROM public.ecr.aws/docker/library/python:3.12-slim

RUN pip install --no-cache-dir \
    "catboost>=1.2" \
    "pandas>=2.0" \
    "numpy>=1.26" \
    "flask>=3.0" \
    "gunicorn>=21.2"

WORKDIR /opt/program
COPY serve.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program

ENTRYPOINT ["./entrypoint.sh"]
