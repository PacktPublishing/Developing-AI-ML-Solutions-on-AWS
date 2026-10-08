# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "scikit-learn", "catboost", "numpy"]
# ///
"""Train the CatBoost fraud classifier the streaming consumer scores with.

Fits on the chronological training split (has_time keeps the rows in time order so no
later transaction informs an earlier one) with balanced class weights, then chooses the
operating threshold by business value with scikit-learn's TunedThresholdClassifierCV:
each blocked fraud, missed fraud (its amount), blocked good card, and approved good card
is priced, and the threshold worth the most on held-out training rows is frozen and
shipped in model_meta.json, the contract the consumer loads.

Usage:
  uv run scoring/train.py
"""

import json
import os

import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, make_scorer, roc_auc_score
from calibration import REVIEW_CUT, to_fpr, to_score
from sklearn.model_selection import TunedThresholdClassifierCV

sklearn.set_config(enable_metadata_routing=True)

CHAPTER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TARGET = "is_fraud"
AMOUNT = "amount_usd"
TIME = "event_time"


def business_metric(y_true, y_pred, amount):
    """Define a business metric for fraud detection."""
    mask_true_positive = (y_true == 1) & (y_pred == 1)
    mask_true_negative = (y_true == 0) & (y_pred == 0)
    mask_false_positive = (y_true == 0) & (y_pred == 1)
    mask_false_negative = (y_true == 1) & (y_pred == 0)
    fraudulent_refuse = mask_true_positive.sum() * 100
    fraudulent_accept = -amount[mask_false_negative].sum()
    legitimate_refuse = mask_false_positive.sum() * -10
    legitimate_accept = (amount[mask_true_negative] * 0.02).sum()
    return fraudulent_refuse + fraudulent_accept + legitimate_refuse + legitimate_accept


def main() -> None:
    """Fit on the past, tune the threshold for business value, evaluate on the future."""
    with open(f"{CHAPTER_DIR}/data/feature_spec.json") as f:
        features = json.load(f)["features"]
    train = pd.read_csv(f"{CHAPTER_DIR}/data/split/train.csv").sort_values(TIME)
    test = pd.read_csv(f"{CHAPTER_DIR}/data/split/test.csv")

    # has_time=True keeps the rows in their chronological order rather than permuting them,
    # so ordered boosting never lets a later transaction inform an earlier one; auto_class_weights
    # handles the imbalance (fraud is a fraction of a percent of rows).
    model = CatBoostClassifier(
        iterations=300,
        auto_class_weights="Balanced",
        has_time=True,
        random_seed=6,
        verbose=0,
    )
    # The classifier is fitted below, inside the threshold search, so that the
    # threshold, the calibration and the shipped model all refer to one fit.

    # Threshold by business value on held-out training rows: price each outcome and keep the
    # operating point worth the most. Frozen and shipped; the consumer never re-derives it.
    # The holdout is chronological, the earlier three quarters of the training window to fit
    # and the latest quarter to score candidates on. A float cv would shuffle instead, which
    # would let later behaviour inform the fitting half and undo has_time above.
    cut = int(len(train) * 0.75)
    split = [(np.arange(cut), np.arange(cut, len(train)))]
    business_scorer = make_scorer(business_metric).set_score_request(amount=True)
    tuned_model = TunedThresholdClassifierCV(
        estimator=model,
        scoring=business_scorer,
        cv=split,
        refit=False,  # keep the classifier fitted on the earlier three quarters
        thresholds=100,
        n_jobs=-1,
    )
    tuned_model.fit(train[features], train[TARGET], amount=train[AMOUNT])
    threshold = float(tuned_model.best_threshold_)

    # The endpoint serves a 0-1000 score, not a probability, so the calibration ships
    # beside the model. Its reference has to come from rows the classifier did not fit
    # on: scores on its own training rows are overconfident, so their percentiles would
    # make a cut look safer than it is. Calibrate on the held-back quarter and carry the
    # business threshold onto that same scale.
    fitted = tuned_model.estimator_
    valid = train.iloc[cut:]
    valid_proba = fitted.predict_proba(valid[features])[:, 1]
    legit = np.sort(valid_proba[valid[TARGET].to_numpy() == 0])
    block_cut = int(to_score([threshold], legit)[0])

    test_scores = fitted.predict_proba(test[features])[:, 1]
    blocked = test_scores >= threshold
    caught = int((blocked & (test[TARGET] == 1)).sum())
    frauds = int((test[TARGET] == 1).sum())
    value = business_metric(
        test[TARGET].to_numpy(), blocked.astype(int), test[AMOUNT].to_numpy()
    )
    print(
        f"score threshold {block_cut} (from probability {threshold:.4f}), "
        f"a {to_fpr(block_cut):.1%} false-alarm rate by construction"
    )
    print(f"test ROC AUC {roc_auc_score(test[TARGET], test_scores):.3f}")
    print(f"test PR  AUC {average_precision_score(test[TARGET], test_scores):.3f}")
    print(
        f"threshold {threshold:.4f}: blocks {int(blocked.sum())} of {len(test)}, "
        f"catches {caught}/{frauds} frauds "
        f"(recall {caught / frauds:.1%}, precision {caught / max(int(blocked.sum()), 1):.1%}), "
        f"business value {value:,.0f}"
    )

    os.makedirs(f"{CHAPTER_DIR}/artifacts", exist_ok=True)
    # ship the classifier the threshold and the calibration refer to
    fitted.save_model(f"{CHAPTER_DIR}/artifacts/model.cbm")
    np.save(f"{CHAPTER_DIR}/artifacts/calibration.npy", legit)
    with open(f"{CHAPTER_DIR}/artifacts/model_meta.json", "w") as f:
        json.dump(
            {
                "features": features,
                "probability_threshold": threshold,
                "block_cut": block_cut,
                "review_cut": REVIEW_CUT,
            },
            f,
            indent=2,
        )
    print("saved artifacts/model.cbm, calibration.npy and model_meta.json")


if __name__ == "__main__":
    main()
