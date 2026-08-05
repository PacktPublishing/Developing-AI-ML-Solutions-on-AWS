"""SageMaker inference server for the fraud model (custom container).

Serves the SageMaker contract on :8080 (GET /ping, POST /invocations), plus the
prefixed /endpoints/<name>/invocations path the boto3 sagemaker-runtime client
posts to, so the same image and the same call work locally and behind
Serverless Inference. Accepts JSON (a record, a list, or {"instances": [...]})
and answers a fraud probability per row.
"""

import json
import os

import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from flask import Flask, Response, request

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
app = Flask(__name__)
_model: CatBoostClassifier | None = None
_features: list[str] | None = None


def _load() -> tuple[CatBoostClassifier, list[str]]:
    """Load the model once, on the first request (SageMaker warms /ping first)."""
    global _model, _features
    if _model is not None and _features is not None:
        return _model, _features
    model = CatBoostClassifier()
    model.load_model(f"{MODEL_DIR}/model.cbm")
    with open(f"{MODEL_DIR}/model_meta.json") as f:
        features = list(json.load(f)["features"])
    _model, _features = model, features
    return model, features


@app.route("/ping", methods=["GET"])
def ping() -> Response:
    """Health check: 200 once the model artifact can be loaded."""
    try:
        _load()
        return Response(status=200)
    except (CatBoostError, OSError, KeyError, ValueError):
        # missing or unreadable artifact, malformed metadata: not ready yet
        return Response(status=503)


@app.route("/invocations", methods=["POST"])
@app.route("/endpoints/<name>/invocations", methods=["POST"])
def invocations(name: str | None = None) -> Response:
    """Score a batch of transactions, returning one probability per row."""
    model, features = _load()
    payload = json.loads(request.data or b"{}")
    records = (
        payload["instances"]
        if isinstance(payload, dict) and "instances" in payload
        else payload
    )
    if isinstance(records, dict):
        records = [records]
    scores = model.predict_proba(pd.DataFrame(records)[features])[:, 1]
    return Response(
        json.dumps({"scores": [round(float(s), 6) for s in scores]}),
        status=200,
        mimetype="application/json",
    )
