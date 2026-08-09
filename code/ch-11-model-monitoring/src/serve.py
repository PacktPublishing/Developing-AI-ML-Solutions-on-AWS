"""SageMaker inference server for the scorecard, for Batch Transform.

GET /ping and POST /invocations on port 8080. It accepts CSV -- headerless feature
rows in the trained column order -- and returns the probability of default per row.
The Batch Transform step scores the current batch through this server; the monitor
then reads those live scores. The same image trains, serves, and monitors.
"""

import io
import os

import pandas as pd
from flask import Flask, Response, request
from model import CATEGORICAL, FEATURES, load

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
app = Flask(__name__)
_model = None


def _load():
    """Load the scorecard once, on the first request (SageMaker warms /ping first)."""
    global _model
    if _model is None:
        _model = load(os.path.join(MODEL_DIR, "scorecard.cbm"))
    return _model


@app.route("/ping", methods=["GET"])
def ping() -> Response:
    """Health check: 200 once the model artifact loads."""
    try:
        _load()
        return Response(status=200)
    except Exception:
        return Response(status=503)


@app.route("/invocations", methods=["POST"])
def invocations() -> Response:
    """Score headerless CSV feature rows, returning one probability of default each."""
    model = _load()
    df = pd.read_csv(
        io.StringIO(request.data.decode("utf-8")), header=None, names=FEATURES
    )
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    proba = model.predict_proba(df[FEATURES])[:, 1]
    return Response(
        "\n".join(f"{p:.6f}" for p in proba), status=200, mimetype="text/csv"
    )
