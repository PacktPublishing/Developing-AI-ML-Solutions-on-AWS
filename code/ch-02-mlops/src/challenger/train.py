#!/usr/bin/env python3
"""SageMaker training entry point for the CatBoost challenger (custom container).

Same training contract and channels as the scorecard, so the identical image runs locally and as a managed SageMaker job:

  /opt/ml/input/data/train/        train.csv + feature_spec.json
  /opt/ml/input/data/validation/   test.csv
  /opt/ml/input/config/hyperparameters.json
  /opt/ml/model/                   the fitted model is written here

Logs to the same MLflow experiment as the incumbent, tagged role=challenger, for a side-by-side AUC comparison.
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
    """AUC, Gini, and KS: the same yardstick the incumbent is measured by."""
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = float(roc_auc_score(y_true, p_default))
    fpr, tpr, _ = roc_curve(y_true, p_default)
    ks = float(max(tpr - fpr))
    return {"auc": auc, "gini": 2 * auc - 1, "ks": ks}


# -------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------
def train() -> None:
    """Fit the monotone CatBoost challenger, evaluate it, log the run, save it."""
    from catboost import CatBoostClassifier, Pool

    hp = _hyperparameters()
    monotonic = str(hp.get("monotonic", "true")).lower() == "true"
    params = dict(
        iterations=int(hp.get("n_estimators", 300)),
        depth=int(hp.get("max_depth", 4)),
        learning_rate=float(hp.get("learning_rate", 0.05)),
    )

    with open(os.path.join(TRAIN, "feature_spec.json")) as fh:
        spec = json.load(fh)
    target = spec["target"]
    numeric, categorical = spec["numeric_features"], spec["categorical_features"]
    features = numeric + categorical

    train_df = pd.read_csv(os.path.join(TRAIN, "train.csv"))
    X = train_df[features].copy()
    for col in categorical:
        X[col] = X[col].astype(str)
    y = train_df[target]

    # The business rule, imposed: the same signed directions the scorecard binning
    # honors, passed to CatBoost as monotone constraints on the numeric features
    # (CatBoost does not constrain categoricals, so they are left out).
    monotone = spec["monotone_constraints"] if monotonic else {f: 0 for f in numeric}
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=1,
        verbose=False,
        monotone_constraints={f: monotone[f] for f in numeric},
        **params,
    )
    model.fit(Pool(X, y, cat_features=categorical))
    challenger = ChallengerModel(model, spec)

    metrics = {}
    valid_path = os.path.join(VALID, "test.csv")
    if os.path.exists(valid_path):
        test_df = pd.read_csv(valid_path)
        metrics = _metrics(test_df[target], challenger.predict_proba(test_df))
        print("validation:", {k: round(v, 4) for k, v in metrics.items()})
        # SageMaker scrapes this printed line from CloudWatch with a regex; the format must match the metric_definitions Regex in aws/jobs/amt.py ("validation_auc: ([0-9.]+)").
        print(f"validation_auc: {metrics['auc']:.6f}")

    challenger.save(MODEL)
    _log_to_mlflow(hp, {**params, "monotonic": monotonic}, metrics, model)
    print(f"challenger written to {MODEL}")


# -------------------------------------------------------------------------------
# Experiment tracking
# -------------------------------------------------------------------------------
def _log_to_mlflow(hp: dict, params: dict, metrics: dict, cb_model) -> None:
    """Best-effort experiment tracking; skipped when no tracking server is set."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLFLOW_TRACKING_URI unset, skipping experiment tracking")
        return
    import mlflow

    experiment = hp.get("mlflow_experiment", "credit-scorecard")
    # Local (sqlite) mode has no server to assign an artifact location, so set one on S3 explicitly; the serverless MLflow App manages its own, so MLFLOW_ARTIFACT_ROOT stays unset there.
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    if artifact_root and mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=f"{artifact_root}/{experiment}"
        )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=hp.get("run_name", "challenger")):
        mlflow.set_tag("model_family", "catboost")
        mlflow.set_tag("role", "challenger")
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        # CatBoost has a native MLflow flavor, so it logs directly -- no custom pyfunc --
        # and registers for promotion from the registry.
        mlflow.catboost.log_model(
            cb_model=cb_model,
            name="model",
            pip_requirements=["catboost", "scikit-learn", "pandas"],
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
