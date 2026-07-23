#!/usr/bin/env python3
"""SageMaker training entry point for the XGBoost challenger (custom container).

Same SageMaker training contract and the same channels as the scorecard, so the
identical image runs as a local training job and as a managed SageMaker job:

  /opt/ml/input/data/train/        train.csv + feature_spec.json
  /opt/ml/input/data/validation/   test.csv
  /opt/ml/input/config/hyperparameters.json
  /opt/ml/model/                   the fitted model is written here

The point of the run is the comparison: it logs to the same MLflow experiment as
the incumbent, tagged role=challenger, so the two sit side by side — the
challenger usually wins on AUC, and the question the chapter asks is whether that
lift is worth giving up the scorecard's transparency, now that both obey the same
monotone rule.
"""

import json
import os
import traceback

import pandas as pd
from challenger_model import ChallengerModel

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
    """Read SageMaker hyperparameters (all values arrive as strings).

    A real training job (AWS, or ModelTrainer's SAGEMAKER_TRAINING_JOB mode) writes
    them to the config file. SageMaker LOCAL-mode ModelTrainer does not write that
    file, so also accept them as a JSON object in the SM_HPS env var; env values
    win, so a caller can drive a local search without a mounted config.
    """
    hp = {}
    if os.path.exists(CONFIG):
        with open(CONFIG) as fh:
            hp = json.load(fh)
    if os.environ.get("SM_HPS"):
        hp = {**hp, **json.loads(os.environ["SM_HPS"])}
    return hp


def _metrics(y_true, p_default) -> dict:
    """AUC, Gini, and KS — the same yardstick the incumbent is measured by."""
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = float(roc_auc_score(y_true, p_default))
    fpr, tpr, _ = roc_curve(y_true, p_default)
    ks = float(max(tpr - fpr))
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks}


# -------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------
def train() -> None:
    """Fit the monotone XGBoost challenger, evaluate it, log the run, save it."""
    import xgboost as xgb

    hp = _hyperparameters()
    monotonic = str(hp.get("monotonic", "true")).lower() == "true"
    params = dict(
        n_estimators=int(hp.get("n_estimators", 300)),
        max_depth=int(hp.get("max_depth", 4)),
        learning_rate=float(hp.get("learning_rate", 0.05)),
        subsample=float(hp.get("subsample", 0.8)),
        colsample_bytree=float(hp.get("colsample_bytree", 0.8)),
    )

    with open(os.path.join(TRAIN, "feature_spec.json")) as fh:
        spec = json.load(fh)
    target = spec["target"]
    numeric, categorical = spec["numeric_features"], spec["categorical_features"]
    features = numeric + categorical

    train_df = pd.read_csv(os.path.join(TRAIN, "train.csv"))
    X = train_df[features].copy()
    for col in categorical:
        X[col] = X[col].astype("category")
    y = train_df[target]

    # The business rule, imposed: the same signed directions the scorecard binning
    # honors, passed to XGBoost as monotone constraints (categoricals are 0).
    monotone = spec["monotone_constraints"] if monotonic else {f: 0 for f in features}
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        monotone_constraints={f: monotone[f] for f in features},
        **params,
    )
    model.fit(X, y)
    challenger = ChallengerModel(model, spec)

    metrics = {}
    valid_path = os.path.join(VALID, "test.csv")
    if os.path.exists(valid_path):
        test_df = pd.read_csv(valid_path)
        metrics = _metrics(test_df[target], challenger.predict_proba(test_df))
        print("validation:", {k: round(v, 4) for k, v in metrics.items()})
        # The metric contract: SageMaker captures training metrics by scraping this
        # printed line from CloudWatch with a regex, not from a return value. This
        # exact format must match the metric_definitions Regex in aws/jobs/amt.py
        # ("validation_auc: ([0-9.]+)"); it is what Automatic Model Tuning optimizes.
        print(f"validation_auc: {metrics['auc']:.6f}")

    challenger.save(MODEL)
    _log_to_mlflow(hp, {**params, "monotonic": monotonic}, metrics)
    print(f"challenger written to {MODEL}")


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
    with mlflow.start_run(run_name=hp.get("run_name", "challenger")):
        mlflow.set_tag("model_family", "xgboost")
        mlflow.set_tag("role", "challenger")
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        from mlflow_pyfunc import ChallengerPyfunc

        mlflow.pyfunc.log_model(
            name="model",
            python_model=ChallengerPyfunc(),
            artifacts={"model_dir": MODEL},
            code_paths=["challenger_model.py", "mlflow_pyfunc.py"],
            pip_requirements=["xgboost", "scikit-learn", "pandas", "sagemaker"],
            registered_model_name=hp.get("registered_model_name", "credit-challenger"),
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
