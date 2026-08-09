# The scorecard's training image: it satisfies the SageMaker training contract, so
# the same image runs in local mode (a container here) and as a managed training job
# on AWS. Composed from source on a slim base -- no deep-learning container to pull;
# in production the built-in image plus a requirements.txt meets the same contract.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "catboost>=1.2" \
    "pandas>=2.0" \
    "numpy>=1.26"

WORKDIR /opt/program
COPY src/model.py src/train.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# /opt/program on PATH so `train` resolves; unbuffered logs stream to CloudWatch
# (AWS) and the console (local) alike.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program \
    PATH="/opt/program:${PATH}"

ENTRYPOINT ["./entrypoint.sh"]
