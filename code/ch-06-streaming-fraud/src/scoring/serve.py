"""SageMaker inference server for the fraud model (custom container).

Serves the SageMaker contract on :8080 (GET /ping, POST /invocations), plus the
prefixed /endpoints/<name>/invocations path the boto3 sagemaker-runtime client
posts to, so the same image and the same call work locally and behind
Serverless Inference. Accepts JSON (a record, a list, or {"instances": [...]})
and answers a 0-1000 fraud score per row: the model's probability put through
the calibration frozen at training time, so the consumer compares a score to a
score and the dashboard reads the same scale the policy is written on.
"""

import json
import os

import numpy as np
import pandas as pd
from calibration import to_score
from catboost import CatBoostClassifier, CatBoostError
from flask import Flask, Response, request

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
app = Flask(__name__)
_model: CatBoostClassifier | None = None
_features: list[str] | None = None
_legit: np.ndarray | None = None


def _load() -> tuple[CatBoostClassifier, list[str], np.ndarray]:
    """Load the model and its calibration once, on the first request."""
    global _model, _features, _legit
    if _model is not None and _features is not None and _legit is not None:
        return _model, _features, _legit
    model = CatBoostClassifier()
    model.load_model(f"{MODEL_DIR}/model.cbm")
    with open(f"{MODEL_DIR}/model_meta.json") as f:
        features = list(json.load(f)["features"])
    legit = np.load(f"{MODEL_DIR}/calibration.npy")
    _model, _features, _legit = model, features, legit
    return model, features, legit


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
    """Score a batch of transactions, returning one 0-1000 score per row."""
    model, features, legit = _load()
    payload = json.loads(request.data or b"{}")
    records = (
        payload["instances"]
        if isinstance(payload, dict) and "instances" in payload
        else payload
    )
    if isinstance(records, dict):
        records = [records]
    proba = model.predict_proba(pd.DataFrame(records)[features])[:, 1]
    scores = to_score(proba, legit)
    return Response(
        json.dumps({"scores": [int(s) for s in scores]}),
        status=200,
        mimetype="application/json",
    )
