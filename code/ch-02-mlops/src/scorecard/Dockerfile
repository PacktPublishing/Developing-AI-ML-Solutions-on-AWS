# The scorecard's custom container: it both trains and serves, so the exact
# image that produced /opt/ml/model is the one behind the endpoint. Composed
# from source on a slim base — no SageMaker deep-learning container to pull. In
# production the same contract is often met with the built-in
# sagemaker-scikit-learn image plus a requirements.txt; this is the transparent
# equivalent.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "fastwoe==0.1.8" \
    "scikit-learn>=1.3" \
    "pandas>=2.0" \
    "numpy>=1.26" \
    "joblib>=1.3" \
    "flask>=3.0" \
    "gunicorn>=21.2" \
    "mlflow>=3.10,<4" \
    "sagemaker-mlflow>=0.1" \
    "boto3>=1.34"

WORKDIR /opt/program
COPY scorecard_model.py train.py serve.py mlflow_pyfunc.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# /opt/program on PATH so `serve`/`train` resolve; unbuffered logs stream to
# CloudWatch (AWS) and the console (local) alike.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program \
    PATH="/opt/program:${PATH}" \
    GIT_PYTHON_REFRESH=quiet

ENTRYPOINT ["./entrypoint.sh"]
