"""Lambda handler that loads the challenger from the MLflow registry at cold start.

The counterpart to the baked-image Lambda: instead of copying the model into the image,
this variant resolves the registered version and downloads its run artifacts at first
invoke (to /tmp, the only writable path in Lambda), rebuilds the ChallengerModel, and
scores warm invocations from it. mlflow-skinny plus the sagemaker-mlflow plugin reach the
registry, on AWS a SageMaker MLflow App by ARN, locally the serverless sqlite store, and
catboost loads the native flavor's raw serving files.

Env: MLFLOW_TRACKING_URI (required), REGISTERED_MODEL (default credit-challenger),
MODEL_VERSION (optional; default the latest registered version).
"""

import json
import os

import mlflow
from challenger_model import ChallengerModel
from mlflow.tracking import MlflowClient


def _load_from_registry() -> ChallengerModel:
    """Resolve the registered version and pull its serving files to /tmp."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    name = os.environ.get("REGISTERED_MODEL", "credit-challenger")
    client = MlflowClient()
    if version := os.environ.get("MODEL_VERSION"):
        mv = client.get_model_version(name, version)
    elif versions := client.search_model_versions(f"name='{name}'"):
        mv = max(versions, key=lambda v: int(v.version))
    else:
        raise RuntimeError(f"no registered versions of {name}")
    # download the run's raw serving files (challenger.cbm + feature_spec.json), then
    # rebuild the wrapper; /tmp is the only writable path in the Lambda filesystem
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, dst_path="/tmp/model"
    )
    print(f"loaded {name} v{mv.version} (run {mv.run_id}) from {local_dir}")
    return ChallengerModel.load(local_dir)


# Cold-start work: resolve the version and pull the artifact once, reuse when warm.
_model = _load_from_registry()


def lambda_handler(event, context):
    """Score a batch of applications and return probability of default per row."""
    body = event.get("body", event) if isinstance(event, dict) else event
    if isinstance(body, str):
        body = json.loads(body)
    records = (
        body["instances"] if isinstance(body, dict) and "instances" in body else body
    )
    if isinstance(records, dict):
        records = [records]
    pd_default = _model.predict_proba(records)
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"pd": pd_default}),
    }
