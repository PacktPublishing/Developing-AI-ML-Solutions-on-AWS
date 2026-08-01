"""Pipeline step (Processing): score the shortlist with the trained model."""

import os
import tarfile

import pandas as pd
from catboost import CatBoostClassifier

IN = "/opt/ml/processing/input"
MODEL_IN = "/opt/ml/processing/model"
OUT = "/opt/ml/processing/output"


def main() -> None:
    """Load the training step's artifact and score the eligible book."""
    features = os.environ["FEATURES"].split(",")
    eligible = pd.read_csv(f"{IN}/eligible.csv")

    # the training step ships model.tar.gz; unpack it to get model.cbm
    for name in os.listdir(MODEL_IN):
        if name.endswith(".tar.gz"):
            with tarfile.open(f"{MODEL_IN}/{name}") as tar:
                tar.extractall(MODEL_IN, filter="data")
    model = CatBoostClassifier()
    model.load_model(f"{MODEL_IN}/model.cbm")

    scored = eligible[["customer_id", "current_limit", "utilization"]].copy()
    scored["pd_12m"] = model.predict_proba(eligible[features])[:, 1].round(6)

    os.makedirs(OUT, exist_ok=True)
    scored.to_csv(f"{OUT}/scores.csv", index=False)
    print(f"scored {len(scored)} customers (mean pd {scored['pd_12m'].mean():.3f})")


if __name__ == "__main__":
    main()
