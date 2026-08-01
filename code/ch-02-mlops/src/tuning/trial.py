#!/usr/bin/env python3
"""One Syne Tune trial: train the monotone XGBoost challenger, report validation_auc.

Syne Tune's LocalBackend runs this as a subprocess per trial, passing the sampled
hyperparameters as CLI flags and reading the objective back through Reporter (not a
stdout regex, the way managed AMT scrapes CloudWatch). The training matches
challenger/train.py exactly — same feature_spec, same monotone_constraints, same
predict_proba[:, 1] — so the local search optimizes what the container would.
"""

import argparse
import json
import os

import pandas as pd
import xgboost as xgb
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
    features = spec["numeric_features"] + spec["categorical_features"]
    monotone = spec["monotone_constraints"]

    def frame(df):
        X = df[features].copy()
        for col in spec["categorical_features"]:
            X[col] = X[col].astype("category")
        return X

    tr = pd.read_csv(f"{a.data_dir}/train/train.csv")
    te = pd.read_csv(f"{a.data_dir}/validation/test.csv")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        monotone_constraints={f: monotone[f] for f in features},
        n_estimators=a.n_estimators,
        max_depth=a.max_depth,
        learning_rate=a.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(frame(tr), tr[target])
    auc = float(roc_auc_score(te[target], model.predict_proba(frame(te))[:, 1]))

    # Same MLflow path as challenger/train.py: point MLFLOW_TRACKING_URI at the App.
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "credit-challenger"))
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
