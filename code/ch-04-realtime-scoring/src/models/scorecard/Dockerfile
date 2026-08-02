# The scorecard model as a SageMaker bring-your-own inference container: the same
# image SageMaker local mode runs on your laptop and a real endpoint runs in the
# cloud. Serves GET /ping and POST /invocations on 8080 with FastAPI/uvicorn.
FROM python:3.12-slim

RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29"

WORKDIR /opt/program
COPY model.py serve.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program \
    PATH="/opt/program:${PATH}"

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]
