# The decision gateway's image: the FastAPI service that orchestrates the model
# endpoints. The same container runs under uvicorn on a laptop and on ECS Fargate
# behind the load balancer; only ENDPOINTS changes — model container URLs locally,
# SageMaker endpoint names in the cloud.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "fastapi>=0.110" \
    "uvicorn>=0.29" \
    "httpx>=0.27" \
    "boto3>=1.34"

WORKDIR /opt/app
COPY app.py ./

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
