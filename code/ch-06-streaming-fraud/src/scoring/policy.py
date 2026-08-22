# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "catboost", "matplotlib", "scikit-learn"]
# ///
"""Decision policy: a 0-1000 fraud score calibrated to Amazon Fraud Detector's table, banded by two cuts.

The score reproduces Amazon Fraud Detector's published score-to-false-positive-rate calibration on
our own model: a record's false-positive rate is measured against the legitimate traffic, then
mapped through AFD's seven anchor points, so a score of 900 means a 2% false-alarm rate, exactly as
AFD's score does. Two cuts split it into approve, investigate, and fraud. FPRCalibrator is a
scikit-learn transformer, so it pickles and drops into a Pipeline after the model.

Usage:
  uv run scoring/policy.py
"""

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------------------------------------
# Amazon Fraud Detector's published score-to-FPR anchors (docs: model-scores.html)
# and the plotting palette and band cuts
# -------------------------------------------------------------------------------
CHAPTER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
AFD_FPR = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10])
AFD_SCORE = np.array([975, 950, 900, 860, 775, 700, 600])
APPROVE = "#2CA25F"
INVESTIGATE = "#F0A030"
FRAUD = "#E4408A"
REVIEW_CUT = 700  # investigate at and above this score (a 7% false-alarm rate)
FRAUD_CUT = 900  # flag as fraud at and above this score (a 2% false-alarm rate)


def decide(
    score: float, review_cut: float = REVIEW_CUT, fraud_cut: float = FRAUD_CUT
) -> str:
    """Band a 0-1000 fraud score into approve, investigate, or fraud. Ordered, first match wins."""
    if score >= fraud_cut:
        return "fraud"
    if score >= review_cut:
        return "investigate"
    return "approve"


def _fpr_to_score(fpr: np.ndarray) -> np.ndarray:
    """Map a false-positive rate to the 0-1000 score through AFD's anchors (the score's definition)."""
    fpr = np.asarray(fpr, float)
    s = np.empty_like(fpr)
    lo = fpr <= AFD_FPR[0]
    s[lo] = 1000 - (fpr[lo] / AFD_FPR[0]) * (1000 - AFD_SCORE[0])
    mid = (fpr > AFD_FPR[0]) & (fpr <= AFD_FPR[-1])
    s[mid] = np.interp(fpr[mid], AFD_FPR, AFD_SCORE)
    hi = fpr > AFD_FPR[-1]
    s[hi] = AFD_SCORE[-1] * (1 - fpr[hi]) / (1 - AFD_FPR[-1])
    return s


class FPRCalibrator(BaseEstimator, TransformerMixin):
    """Map a probability of fraud to a 0-1000 score calibrated to Amazon Fraud Detector's table.

    fit freezes the legitimate-probability reference from the training sample; transform measures
    each probability's false-positive rate against that reference (the share of legitimate records
    it outranks) and maps it through AFD's seven anchor points, so a score of 900 is a 2% false-alarm
    rate, reproducing AFD's calibration on our own model. Fitted instances pickle, so the calibration
    ships with the model.
    """

    def fit(self, proba, y):
        """Freeze the sorted legitimate probabilities from the training scores and labels."""
        proba = np.asarray(proba, float).ravel()
        y = np.asarray(y).ravel()
        self.legit_ = np.sort(proba[y == 0])
        return self

    def transform(self, proba) -> np.ndarray:
        """Score each probability: its false-positive rate against the reference, mapped through AFD's anchors."""
        proba = np.asarray(proba, float).ravel()
        fpr = 1 - np.searchsorted(self.legit_, proba, side="left") / len(self.legit_)
        return np.round(_fpr_to_score(fpr)).astype(int).reshape(-1, 1)


class ProbaExtractor(BaseEstimator, TransformerMixin):
    """Turn a fitted probabilistic classifier into a transformer: features in, P(fraud) out.

    Lets a classifier sit before FPRCalibrator in a Pipeline, e.g.
    make_pipeline(ProbaExtractor(model), FPRCalibrator()).fit(X_train, y_train). fit is a no-op.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        """No-op: the wrapped classifier is already fitted."""
        return self

    def transform(self, X) -> np.ndarray:
        """Return P(fraud) as a column so a downstream transformer receives a 2-D array."""
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)


def main() -> None:
    """Score the test rows with the AFD-calibrated score, band them, report the split, plot it."""
    with open(f"{CHAPTER_DIR}/artifacts/model_meta.json") as f:
        features = json.load(f)["features"]
    model = CatBoostClassifier()
    model.load_model(f"{CHAPTER_DIR}/artifacts/model.cbm")
    train = pd.read_csv(f"{CHAPTER_DIR}/data/split/train.csv")
    test = pd.read_csv(f"{CHAPTER_DIR}/data/split/test.csv")

    calibrator = FPRCalibrator().fit(
        model.predict_proba(train[features])[:, 1], train["is_fraud"].values
    )
    score = calibrator.transform(model.predict_proba(test[features])[:, 1]).ravel()
    outcome = np.array([decide(s) for s in score])
    counts = {o: int((outcome == o).sum()) for o in ("approve", "investigate", "fraud")}
    print(f"review_cut {REVIEW_CUT} (7% FPR), fraud_cut {FRAUD_CUT} (2% FPR): {counts}")

    plt.rcParams.update({"font.family": "Arial", "font.size": 15})
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    bins = np.linspace(0, 1000, 60)
    for lo, hi, color in [
        (0, REVIEW_CUT, APPROVE),
        (REVIEW_CUT, FRAUD_CUT, INVESTIGATE),
        (FRAUD_CUT, 1001, FRAUD),
    ]:
        ax.hist(score[(score >= lo) & (score < hi)], bins=bins, color=color)
    for x in (REVIEW_CUT, FRAUD_CUT):
        ax.axvline(x, color="#444", ls="--", lw=1.4)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
    ax.legend(
        handles=[
            mpatches.Patch(color=APPROVE, label=f"approve {counts['approve']:,}"),
            mpatches.Patch(
                color=INVESTIGATE,
                label=f"investigate {counts['investigate']:,} (>= {REVIEW_CUT}, a 7% false-alarm rate)",
            ),
            mpatches.Patch(
                color=FRAUD,
                label=f"fraud {counts['fraud']:,} (>= {FRAUD_CUT}, a 2% false-alarm rate)",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=15,
    )
    ax.set_xlim(0, 1000)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Fraud score (0-1000)", fontsize=16)
    ax.set_ylabel("Transactions", fontsize=16)
    ax.set_title("Fraud score distribution, with its decision bands", fontsize=18)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    os.makedirs(f"{CHAPTER_DIR}/artifacts", exist_ok=True)
    plt.savefig(f"{CHAPTER_DIR}/artifacts/score_bands.png")
    print("saved artifacts/score_bands.png")


if __name__ == "__main__":
    main()
