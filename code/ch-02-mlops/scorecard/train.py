#!/usr/bin/env python3
"""SageMaker training entry point for the WOE scorecard (custom container).

Implements the SageMaker training contract, so the same image runs unchanged as
a local training job (docker run ... train) and as a managed SageMaker training
job on real AWS:

  /opt/ml/input/data/train/        train.csv + feature_spec.json
  /opt/ml/input/data/validation/   test.csv  (held-out, for honest metrics)
  /opt/ml/input/config/hyperparameters.json
  /opt/ml/model/                   the fitted model is written here
  /opt/ml/output/failure           a readable reason if training fails

Experiment tracking is best-effort: if MLFLOW_TRACKING_URI is set (the local
MLflow server, or a SageMaker managed MLflow ARN on AWS) the run's params,
metrics, and artifacts are logged and the model is registered; if not, training
still writes /opt/ml/model so the container works standalone.
"""

import json
import os
import traceback

import pandas as pd
from scorecard_model import ScorecardPredictor

# -------------------------------------------------------------------------------
# The SageMaker path contract
# -------------------------------------------------------------------------------
PREFIX = "/opt/ml"
TRAIN = f"{PREFIX}/input/data/train"
VALID = f"{PREFIX}/input/data/validation"
CONFIG = f"{PREFIX}/input/config/hyperparameters.json"
MODEL = f"{PREFIX}/model"
FAILURE = f"{PREFIX}/output/failure"


# -------------------------------------------------------------------------------
# Hyperparameters and metrics
# -------------------------------------------------------------------------------
def _hyperparameters() -> dict:
    """Read SageMaker hyperparameters (all values arrive as strings)."""
    if os.path.exists(CONFIG):
        with open(CONFIG) as fh:
            return json.load(fh)
    return {}


def _metrics(y_true, p_default) -> dict:
    """AUC, Gini, and KS — the numbers a credit team reads first."""
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = float(roc_auc_score(y_true, p_default))
    fpr, tpr, _ = roc_curve(y_true, p_default)
    ks = float(max(tpr - fpr))
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks}


# -------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------
def train() -> None:
    """Fit the WOE scorecard, evaluate it, log the run, and save the model."""
    from fastwoe import FastWoe
    from sklearn.linear_model import LogisticRegression

    hp = _hyperparameters()
    C = float(hp.get("C", 1.0))
    max_iter = int(hp.get("max_iter", 1000))
    monotonic = str(hp.get("monotonic", "true")).lower() == "true"

    with open(os.path.join(TRAIN, "feature_spec.json")) as fh:
        spec = json.load(fh)
    target = spec["target"]
    features = spec["numeric_features"] + spec["categorical_features"]

    train_df = pd.read_csv(os.path.join(TRAIN, "train.csv"))
    X, y = train_df[features], train_df[target]

    # The business rule: monotone WOE binning for the constrained numerics. Pass
    # the whole dict (categoricals and unconstrained features are 0 = ignored).
    monotone = spec["monotone_constraints"] if monotonic else {}
    woe = FastWoe(monotonic_cst=monotone or None)
    Xw = woe.fit_transform(X, y)
    lr = LogisticRegression(C=C, max_iter=max_iter).fit(Xw, y)
    model = ScorecardPredictor(woe, lr, spec)

    # Honest metrics on the held-out validation channel when present.
    metrics = {}
    valid_path = os.path.join(VALID, "test.csv")
    if os.path.exists(valid_path):
        test_df = pd.read_csv(valid_path)
        metrics = _metrics(test_df[target], model.predict_proba(test_df))
        print("validation:", {k: round(v, 4) for k, v in metrics.items()})
        # a line SageMaker Automatic Model Tuning parses as the objective metric
        print(f"validation_auc: {metrics['auc']:.6f}")

    model.save(MODEL)

    # Weight-of-Evidence information value per feature — a scorecard deliverable.
    try:
        iv = woe.get_iv_analysis()
        iv.to_csv(os.path.join(MODEL, "iv_analysis.csv"), index=False)
    except Exception as exc:
        print(f"iv_analysis skipped: {exc}")

    _log_to_mlflow(hp, {"C": C, "max_iter": max_iter, "monotonic": monotonic}, metrics)
    print(f"scorecard written to {MODEL}")


# -------------------------------------------------------------------------------
# Experiment tracking
# -------------------------------------------------------------------------------
def _log_to_mlflow(hp: dict, params: dict, metrics: dict) -> None:
    """Best-effort experiment tracking; skipped when no tracking server is set."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLFLOW_TRACKING_URI unset — skipping experiment tracking")
        return
    import mlflow

    # Serverless MLflow has no server, so the client writes artifacts straight to
    # S3: the experiment must be created with an S3 artifact_location the first
    # time (locally the S3Proxy bucket, on AWS a real bucket). If it already
    # exists, set_experiment just selects it.
    experiment = hp.get("mlflow_experiment", "credit-scorecard")
    # Local (sqlite) mode has no server to assign an artifact location, so we set
    # one on S3 explicitly. The serverless MLflow App manages its own artifact
    # store, so there MLFLOW_ARTIFACT_ROOT is left unset and the App decides.
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    if artifact_root and mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=f"{artifact_root}/{experiment}"
        )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=hp.get("run_name", "scorecard")):
        mlflow.set_tag("model_family", "woe-logistic-regression")
        mlflow.set_tag("role", "incumbent")
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if os.path.exists(os.path.join(MODEL, "iv_analysis.csv")):
            mlflow.log_artifact(os.path.join(MODEL, "iv_analysis.csv"))
        # Register the scorecard as a pyfunc model so SageMaker ModelBuilder can
        # later deploy it straight from the registry.
        from mlflow_pyfunc import ScorecardPyfunc

        mlflow.pyfunc.log_model(
            name="model",
            python_model=ScorecardPyfunc(),
            artifacts={"model_dir": MODEL},
            code_paths=["scorecard_model.py", "mlflow_pyfunc.py"],
            pip_requirements=[
                "fastwoe",
                "scikit-learn",
                "pandas",
                "joblib",
                "sagemaker",
            ],
            registered_model_name=hp.get("registered_model_name", "credit-scorecard"),
        )
    print(f"logged run to {uri}")


# -------------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        train()
    except Exception:
        os.makedirs(os.path.dirname(FAILURE), exist_ok=True)
        with open(FAILURE, "w") as fh:
            fh.write(traceback.format_exc())
        raise
