"""SageMaker inference server for the XGBoost challenger (custom container).

Same two-route contract as the scorecard — GET /ping, POST /invocations on port
8080 — so the same image serves a local container, a real-time endpoint, and a
batch-transform job, and the challenger is invoked exactly like the incumbent.
"""

import io
import json
import os

import pandas as pd
from flask import Flask, Response, request

from challenger_model import ChallengerModel

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
app = Flask(__name__)
_model: ChallengerModel | None = None


def _load() -> ChallengerModel:
    """Load the model once, on the first request."""
    global _model
    if _model is None:
        _model = ChallengerModel.load(MODEL_DIR)
    return _model


@app.route("/ping", methods=["GET"])
def ping() -> Response:
    """Health check: 200 once the model artifact can be loaded."""
    try:
        _load()
        return Response(status=200)
    except Exception:
        return Response(status=503)


@app.route("/invocations", methods=["POST"])
def invocations() -> Response:
    """Score a batch of applications, echoing the caller's content type."""
    model = _load()
    content_type = request.content_type or "application/json"

    if content_type.startswith("text/csv"):
        df = pd.read_csv(io.StringIO(request.data.decode("utf-8")))
        pd_default = model.predict_proba(df)
        body = "\n".join(f"{p:.6f}" for p in pd_default)
        return Response(body, status=200, mimetype="text/csv")

    payload = json.loads(request.data or b"{}")
    records = (
        payload["instances"]
        if isinstance(payload, dict) and "instances" in payload
        else payload
    )
    if isinstance(records, dict):
        records = [records]
    pd_default = model.predict_proba(records)
    return Response(
        json.dumps({"pd": pd_default}), status=200, mimetype="application/json"
    )
