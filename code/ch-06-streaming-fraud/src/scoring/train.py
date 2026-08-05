# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "scikit-learn", "catboost"]
# ///
"""Train the CatBoost fraud classifier the streaming consumer scores with.

CatBoost handles the heterogeneous features here (amounts, counts, rates, and
flags in one vector) with little tuning and scores a single row fast enough for
a real-time budget. The trainer reads the chronological split, fits on the past
with balanced class weights (fraud is a fraction of a percent of rows), and
evaluates on the future at the production fraud ratio.

The operating threshold is chosen on the training data, never the test data:
the highest-recall point that still keeps training precision at or above the
target. The default holds precision near 30%, roughly three false alarms for
every fraud caught, because a blocked card is a recoverable annoyance and a
fraudulent charge is not.

Artifacts, under artifacts/: model.cbm (CatBoost's native format, the file a
SageMaker model tarball would carry) and model_meta.json (the feature list and
threshold the consumer must load, the contract between training and serving).

Usage:
  uv run scoring/train.py --min-precision 0.30
"""

import argparse
import json
import os

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

CHAPTER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TARGET = "is_fraud"


def main() -> None:
    """Fit on the past, pick the threshold, evaluate on the future."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-precision", type=float, default=0.30)
    args = parser.parse_args()

    with open(f"{CHAPTER_DIR}/data/feature_spec.json") as f:
        features = json.load(f)["features"]
    train = pd.read_csv(f"{CHAPTER_DIR}/data/split/train.csv")
    test = pd.read_csv(f"{CHAPTER_DIR}/data/split/test.csv")

    # auto_class_weights handles the imbalance: at a fraction of a percent fraud,
    # an unweighted fit would just say "legit" and be right 99.5% of the time.
    model = CatBoostClassifier(
        iterations=300,
        auto_class_weights="Balanced",
        random_seed=6,
        verbose=0,
    )
    model.fit(train[features], train[TARGET])

    # Threshold from the training data: smallest score clearing the target
    # precision. Frozen and shipped with the model; the consumer never re-derives it.
    train_scores = model.predict_proba(train[features])[:, 1]
    precision, _, thresholds = precision_recall_curve(train[TARGET], train_scores)
    ok = precision[:-1] >= args.min_precision
    if not ok.any():
        raise SystemExit(f"no threshold reaches precision {args.min_precision}")
    threshold = float(thresholds[ok.argmax()])

    test_scores = model.predict_proba(test[features])[:, 1]
    blocked = test_scores >= threshold
    caught = (blocked & (test[TARGET] == 1)).sum()
    frauds = int((test[TARGET] == 1).sum())
    print(f"test ROC AUC {roc_auc_score(test[TARGET], test_scores):.3f}")
    print(f"test PR  AUC {average_precision_score(test[TARGET], test_scores):.3f}")
    print(
        f"threshold {threshold:.4f}: blocks {blocked.sum()} of {len(test)}, "
        f"catches {caught}/{frauds} frauds "
        f"(recall {caught / frauds:.1%}, precision {caught / max(blocked.sum(), 1):.1%})"
    )

    os.makedirs(f"{CHAPTER_DIR}/artifacts", exist_ok=True)
    model.save_model(f"{CHAPTER_DIR}/artifacts/model.cbm")
    with open(f"{CHAPTER_DIR}/artifacts/model_meta.json", "w") as f:
        json.dump({"features": features, "threshold": threshold}, f, indent=2)
    print("saved artifacts/model.cbm and artifacts/model_meta.json")


if __name__ == "__main__":
    main()
