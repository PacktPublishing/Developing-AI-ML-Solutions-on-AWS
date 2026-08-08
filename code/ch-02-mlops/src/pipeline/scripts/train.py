"""Pipeline step 2 (Processing): train the scorecard or the challenger.

--model scorecard  -> logistic regression (the transparent incumbent)
--model challenger -> monotone XGBoost (+util, +dpd, -income)
Writes metrics.json (model, auc, held-out scores) to the processing output.
"""

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN = "/opt/ml/processing/input/train.csv"
TEST = "/opt/ml/processing/test/test.csv"
OUT = "/opt/ml/processing/model"
FEATURES = ["util", "dpd", "income"]
TARGET = "default"


def main() -> None:
    """Fit the requested model, score the held-out split, write metrics.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["scorecard", "challenger"])
    args, _ = parser.parse_known_args()

    tr = pd.read_csv(TRAIN)
    te = pd.read_csv(TEST)
    os.makedirs(OUT, exist_ok=True)

    if args.model == "scorecard":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000).fit(tr[FEATURES], tr[TARGET])
    else:
        import xgboost as xgb

        clf = xgb.XGBClassifier(
            tree_method="hist",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            monotone_constraints=(1, 1, -1),
        ).fit(tr[FEATURES], tr[TARGET])

    scores = clf.predict_proba(te[FEATURES])[:, 1]
    auc = float(roc_auc_score(te[TARGET], scores))
    with open(f"{OUT}/metrics.json", "w") as fh:
        json.dump(
            {"model": args.model, "auc": round(auc, 4), "scores": scores.tolist()}, fh
        )
    print(f"train {args.model} -> validation_auc: {auc:.6f}")


if __name__ == "__main__":
    main()
