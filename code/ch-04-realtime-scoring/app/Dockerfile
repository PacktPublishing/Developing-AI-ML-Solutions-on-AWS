# The decision service's image: the same container runs under uvicorn on a
# laptop, on ECS Fargate behind the load balancer, and nowhere does the code
# know which. Composed from source on a slim base.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "fastapi>=0.110" \
    "uvicorn>=0.29" \
    "boto3>=1.34"

WORKDIR /opt/app
COPY app.py scorecard.py ./

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
