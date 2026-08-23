# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "catboost", "matplotlib", "scikit-learn"]
# ///
"""Decision policy: a 0-1000 fraud score from a single calibration, banded by two cuts.

The score is a record's percentile among legitimate traffic, raised to a power: score =
1000 * r**gamma, where r is the share of legitimate records the transaction outranks. gamma
is a shape knob. At gamma = 1 the score is linear, so a cut carries FPR = 1 - score/1000. The
default gamma = ln(0.9)/ln(0.98) puts a 2% false-positive rate at score 900, which is Amazon
Fraud Detector's published anchor, and reproduces the rest of AFD's score-to-FPR table to within
a few points. Either way a cut's false-alarm rate is exact and invertible (score_to_fpr), and it
holds on held-out data because r is uniform on legitimates by construction. FPRCalibrator is a
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
# Calibration knob, plotting palette, and band cuts
# -------------------------------------------------------------------------------
CHAPTER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# gamma that puts a 2% FPR at score 900, Amazon Fraud Detector's canonical anchor
# (docs: model-scores.html); gamma = 1 would give the linear score, FPR = 1 - score/1000.
GAMMA = np.log(0.9) / np.log(0.98)
APPROVE = "#3FA45B"  # AWS-slide green
INVESTIGATE = "#ED9A2E"  # AWS-slide orange
FRAUD = "#2699F0"  # AWS-slide blue
REVIEW_CUT = 700  # investigate at and above this score (about a 7% false-alarm rate)
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


def score_to_fpr(score, gamma: float = GAMMA) -> np.ndarray:
    """Return the false-positive rate a score cut carries: FPR = 1 - (score/1000)**(1/gamma)."""
    return 1 - (np.asarray(score, float) / 1000) ** (1 / gamma)


class FPRCalibrator(BaseEstimator, TransformerMixin):
    """Map a probability of fraud to a 0-1000 score: score = 1000 * r**gamma.

    fit freezes the legitimate-probability reference from the training sample. transform measures
    r, the share of that reference each probability outranks (its percentile among legitimates),
    and raises it to gamma. r is uniform on legitimates by construction, so a cut carries an exact,
    invertible false-alarm rate (score_to_fpr) that holds out of sample. The default gamma puts a
    2% FPR at score 900, matching Amazon Fraud Detector's published anchor; gamma = 1 is the linear
    score. Fitted instances pickle, so the calibration ships with the model.
    """

    def __init__(self, gamma: float = GAMMA):
        self.gamma = gamma

    def fit(self, proba, y):
        """Freeze the sorted legitimate probabilities from the training scores and labels."""
        proba = np.asarray(proba, float).ravel()
        y = np.asarray(y).ravel()
        self.legit_ = np.sort(proba[y == 0])
        return self

    def transform(self, proba) -> np.ndarray:
        """Score each probability: its percentile among the legitimate reference, raised to gamma."""
        proba = np.asarray(proba, float).ravel()
        r = np.searchsorted(self.legit_, proba, side="right") / len(self.legit_)
        return np.round(1000 * r**self.gamma).astype(int).reshape(-1, 1)


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
    """Score the test rows, band them, report the split and each cut's measured FPR, plot it."""
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
    # the false-alarm rate each cut actually costs on the held-out legitimates
    legit = score[test["is_fraud"].values == 0]
    review_fpr = (legit >= REVIEW_CUT).mean()
    fraud_fpr = (legit >= FRAUD_CUT).mean()
    print(f"gamma {GAMMA:.3f}: {counts}")
    print(
        f"  measured FPR: review_cut {REVIEW_CUT} -> {review_fpr:.1%} "
        f"(claimed {score_to_fpr(REVIEW_CUT):.1%}), "
        f"fraud_cut {FRAUD_CUT} -> {fraud_fpr:.1%} (claimed {score_to_fpr(FRAUD_CUT):.1%})"
    )

    plt.rcParams.update({"font.family": "Arial", "font.size": 15})
    _, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    bins = np.linspace(0, 1000, 60)
    for lo, hi, color in [
        (0, REVIEW_CUT, APPROVE),
        (REVIEW_CUT, FRAUD_CUT, INVESTIGATE),
        (FRAUD_CUT, 1001, FRAUD),
    ]:
        ax.hist(
            score[(score >= lo) & (score < hi)],
            bins=bins,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
    for x in (REVIEW_CUT, FRAUD_CUT):
        ax.axvline(x, color="#444", ls="--", lw=1.4)
    ax.set_ylim(
        0, np.histogram(score, bins)[0].max() * 1.32
    )  # headroom for the upper-left legend
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
        fontsize=14,
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
