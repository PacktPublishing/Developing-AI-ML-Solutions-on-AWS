# The scorecard's container: it trains the model, serves it for Batch Transform, and
# runs the drift monitors -- so the same image runs in local mode (a container here)
# and on AWS as a training job, a transform, and a Processing job. Composed from source
# on a slim base -- no deep-learning container to pull; in production the built-in image
# plus a requirements.txt meets the same contract.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "catboost>=1.2" \
    "pandas>=2.0" \
    "numpy>=1.26" \
    "boto3>=1.34" \
    "flask>=3.0" \
    "gunicorn>=21.2"

WORKDIR /opt/program
COPY src/model.py src/train.py src/serve.py src/psi.py src/monitor.py \
    src/cloudwatch.py src/run_monitor.py src/sm_monitor.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# /opt/program on PATH so `train` resolves; unbuffered logs stream to CloudWatch
# (AWS) and the console (local) alike.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program \
    PATH="/opt/program:${PATH}"

ENTRYPOINT ["./entrypoint.sh"]
