#!/usr/bin/env python3
"""One Syne Tune trial: train the monotone CatBoost challenger, report validation_auc.

LocalBackend runs this as a subprocess per trial, passing the sampled hyperparameters as CLI flags and reading the objective back through Reporter (not a stdout regex, the way managed AMT scrapes CloudWatch). The training matches challenger/train.py exactly (CatBoost, same feature_spec, monotone constraints on the numeric features), so the local search optimizes what the container would.
"""

import argparse
import json
import os

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from syne_tune import Reporter


def main() -> None:
    """Train the challenger with the trial's hyperparameters and report its AUC."""
    p = argparse.ArgumentParser()
    p.add_argument("--max_depth", type=int, required=True)
    p.add_argument("--n_estimators", type=int, required=True)
    p.add_argument("--learning_rate", type=float, required=True)
    p.add_argument("--data_dir", required=True)
    a, _ = p.parse_known_args()

    spec = json.load(open(f"{a.data_dir}/train/feature_spec.json"))
    target = spec["target"]
    numeric, categorical = spec["numeric_features"], spec["categorical_features"]
    features = numeric + categorical
    monotone = spec["monotone_constraints"]

    def frame(df):
        X = df[features].copy()
        for col in categorical:
            X[col] = X[col].astype(str)
        return X

    tr = pd.read_csv(f"{a.data_dir}/train/train.csv")
    te = pd.read_csv(f"{a.data_dir}/validation/test.csv")

    # CatBoost takes the CLI flags as iterations/depth/learning_rate, and monotone
    # constraints on the numeric features (categoricals are unconstrained), exactly as
    # challenger/train.py builds the model.
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=1,
        verbose=False,
        monotone_constraints={f: monotone[f] for f in numeric},
        iterations=a.n_estimators,
        depth=a.max_depth,
        learning_rate=a.learning_rate,
    )
    model.fit(Pool(frame(tr), tr[target], cat_features=categorical))
    auc = float(roc_auc_score(te[target], model.predict_proba(frame(te))[:, 1]))

    # Same MLflow path as challenger/train.py: point MLFLOW_TRACKING_URI at the App.
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        import mlflow

        mlflow.set_tracking_uri(uri)
        # The serverless MLflow App 404s get-by-name for a missing experiment, so create it
        # first (ignoring "already exists") rather than letting set_experiment error on it.
        experiment = os.environ.get("MLFLOW_EXPERIMENT", "credit-challenger")
        try:
            mlflow.create_experiment(experiment)
        except mlflow.exceptions.MlflowException:
            pass  # already exists
        mlflow.set_experiment(experiment)
        with mlflow.start_run():
            mlflow.set_tag("role", "challenger")
            mlflow.set_tag("tuner", "syne-tune")
            mlflow.log_params(
                {
                    "max_depth": a.max_depth,
                    "n_estimators": a.n_estimators,
                    "learning_rate": a.learning_rate,
                }
            )
            mlflow.log_metric("validation_auc", auc)

    Reporter()(validation_auc=auc)


if __name__ == "__main__":
    main()
