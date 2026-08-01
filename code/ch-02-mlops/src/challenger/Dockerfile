# The challenger's custom container: trains and serves the monotone XGBoost
# model. Same shape as the scorecard image (one image trains and serves, so the
# runtime that produced the model is the one behind the endpoint) — only the
# library changes, xgboost in place of fastwoe.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "xgboost>=2.0" \
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
COPY challenger_model.py train.py serve.py mlflow_pyfunc.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/program \
    PATH="/opt/program:${PATH}" \
    GIT_PYTHON_REFRESH=quiet

ENTRYPOINT ["./entrypoint.sh"]
