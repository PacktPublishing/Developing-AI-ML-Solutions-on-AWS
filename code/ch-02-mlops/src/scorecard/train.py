#!/usr/bin/env python3
"""SageMaker training entry point for the WOE scorecard (custom container).

Implements the SageMaker training contract, so the same image runs unchanged locally and as a managed SageMaker job:

  /opt/ml/input/data/train/        train.csv + feature_spec.json
  /opt/ml/input/data/validation/   test.csv  (held-out, for honest metrics)
  /opt/ml/input/config/hyperparameters.json
  /opt/ml/model/                   the fitted model is written here
  /opt/ml/output/failure           a readable reason if training fails

Experiment tracking is best-effort: if MLFLOW_TRACKING_URI is set the run is logged and the model registered; if not, training still writes /opt/ml/model.
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
    """AUC, Gini, and KS: the numbers a credit team reads first."""
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
    from sklearn.pipeline import Pipeline

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

    # The scorecard is a scikit-learn pipeline: FastWoe (monotone WOE binning for the
    # constrained numerics carries the business rule) into logistic regression.
    monotone = spec["monotone_constraints"] if monotonic else {}
    pipeline = Pipeline(
        [
            ("woe", FastWoe(monotonic_cst=monotone or None)),
            ("lr", LogisticRegression(C=C, max_iter=max_iter)),
        ]
    )
    pipeline.fit(X, y)
    model = ScorecardPredictor(pipeline, spec)

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

    # Weight-of-Evidence information value per feature, a scorecard deliverable.
    try:
        iv = pipeline.named_steps["woe"].get_iv_analysis()
        iv.to_csv(os.path.join(MODEL, "iv_analysis.csv"), index=False)
    except Exception as exc:
        print(f"iv_analysis skipped: {exc}")

    _log_to_mlflow(
        hp, {"C": C, "max_iter": max_iter, "monotonic": monotonic}, metrics, pipeline
    )
    print(f"scorecard written to {MODEL}")


# -------------------------------------------------------------------------------
# Experiment tracking
# -------------------------------------------------------------------------------
def _log_to_mlflow(hp: dict, params: dict, metrics: dict, pipeline) -> None:
    """Best-effort experiment tracking; skipped when no tracking server is set."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLFLOW_TRACKING_URI unset, skipping experiment tracking")
        return
    import mlflow

    # Create the experiment before selecting it. The serverless MLflow App returns a bare
    # 404 from get-by-name for a missing experiment (which set_experiment surfaces as an
    # error rather than "create it"), so create it explicitly and ignore "already exists".
    # Locally (sqlite), the App manages no artifact store, so set an S3 artifact_location;
    # on the App (MLFLOW_ARTIFACT_ROOT unset) it manages its own.
    experiment = hp.get("mlflow_experiment", "credit-scorecard")
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    create_kwargs = (
        {"artifact_location": f"{artifact_root}/{experiment}"} if artifact_root else {}
    )
    try:
        mlflow.create_experiment(experiment, **create_kwargs)
    except mlflow.exceptions.MlflowException:
        pass  # already exists
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=hp.get("run_name", "scorecard")):
        mlflow.set_tag("model_family", "woe-logistic-regression")
        mlflow.set_tag("role", "incumbent")
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if os.path.exists(os.path.join(MODEL, "iv_analysis.csv")):
            mlflow.log_artifact(os.path.join(MODEL, "iv_analysis.csv"))
        # The scorecard is a scikit-learn pipeline, so it logs with the native sklearn
        # flavor -- no custom pyfunc -- and registers for promotion from the registry.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            pip_requirements=["fastwoe", "scikit-learn", "pandas", "joblib"],
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
