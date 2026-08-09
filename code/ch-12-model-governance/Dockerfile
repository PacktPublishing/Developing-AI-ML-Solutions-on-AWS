# The rollout gateway as one image: the FastAPI app under uvicorn, run unchanged
# locally and as a Fargate service. The from-source AppConfig seam and the router
# live under src/; the feature-flag document is mounted in (or fetched from AppConfig)
# so a rollout widening is a flag edit, not a new image.
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" boto3

COPY src/ /app/src/
COPY app/ /app/app/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
