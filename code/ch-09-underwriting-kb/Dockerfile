# The underwriter app as one image: FastAPI under uvicorn, run unchanged locally
# and in Lambda. The AWS Lambda Web Adapter (the /opt/extensions layer) bridges
# the Lambda runtime to the HTTP server, so there is no handler and no Mangum glue
# -- the same `uvicorn` process serves both worlds.
FROM public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 AS adapter

FROM python:3.12-slim
COPY --from=adapter /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app
# Only the OpenSearch path is used in the cloud, so the image stays lean: no
# psycopg2 (pgvector) and no ollama (local models).
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" boto3 opensearch-py

COPY src/ ./src/
COPY app/ ./app/

ENV PYTHONPATH=/app/src \
    PORT=8080 \
    AWS_LWA_PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
