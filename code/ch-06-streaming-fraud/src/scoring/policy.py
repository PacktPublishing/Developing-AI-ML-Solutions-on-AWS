# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "catboost", "matplotlib"]
# ///
"""Decision policy: turn one fraud score into three outcomes with two cuts.

A single threshold gives block or pass. A second cut adds an investigate band, the
gray zone a human reviews, and that band is the analyst caseload the incidence model
sizes. The rules are ordered and first-match, the same shape a managed rule engine
(such as the now-retired Amazon Fraud Detector) expresses; here it is a few lines.

Usage:
  uv run scoring/policy.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

CHAPTER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
BLOCK_CUT = 0.5  # above this, block outright at the terminal
APPROVE = "#2CA25F"
INVESTIGATE = "#F0A030"
BLOCK = "#E4408A"


def decide(score: float, review_cut: float, block_cut: float = BLOCK_CUT) -> str:
    """Map a fraud score to an outcome. Ordered rules, first match wins."""
    if score >= block_cut:
        return "block"
    if score >= review_cut:
        return "investigate"
    return "approve"


def fpr_score(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a 0-1000 score whose cut carries a *linear* false-alarm meaning.

    Score is how many legitimate records a record outranks, score = 1000 * P(legit
    probability <= this), so FPR = 1 - cut/1000 (a cut at 900 tolerates a 10% FPR). This
    shares the principle of an FPR-anchored score but NOT Amazon Fraud Detector's specific
    curve, which is nonlinear (AFD's 900 is a 2% FPR). Use afd_score to reproduce AFD's
    published calibration.
    """
    legit = np.sort(prob[y == 0])
    ecdf = np.searchsorted(legit, prob, side="right") / len(legit)
    return np.round(1000 * ecdf).astype(int)


# Amazon Fraud Detector's published score-to-FPR anchors for OFI/TFI models
# (docs: model-scores.html). The score is calibrated to the false-positive rate.
AFD_FPR = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10])
AFD_SCORE = np.array([975, 950, 900, 860, 775, 700, 600])


def afd_score(prob: np.ndarray, legit_prob: np.ndarray) -> np.ndarray:
    """Return a 0-1000 score calibrated to Amazon Fraud Detector's published score-to-FPR table.

    A cut then means what it means in AFD (score 900 approx 2% false-alarm rate). For each
    record, the false-positive rate at its probability is measured against a reference set
    of legitimate probabilities, then mapped through AFD's seven anchor points; outside the
    published range (FPR < 0.5% or > 10%) the score extends linearly to 1000 and to 0. On
    held-out data this reproduces AFD's table within the train/test generalization gap.
    """
    legit = np.sort(legit_prob)
    fpr = 1 - np.searchsorted(legit, prob, side="left") / len(legit)  # P(legit >= prob)
    s = np.empty(len(prob))
    lo = fpr <= AFD_FPR[0]
    s[lo] = 1000 - (fpr[lo] / AFD_FPR[0]) * (1000 - AFD_SCORE[0])
    mid = (fpr > AFD_FPR[0]) & (fpr <= AFD_FPR[-1])
    s[mid] = np.interp(fpr[mid], AFD_FPR, AFD_SCORE)
    hi = fpr > AFD_FPR[-1]
    s[hi] = AFD_SCORE[-1] * (1 - fpr[hi]) / (1 - AFD_FPR[-1])
    return np.round(s).astype(int)


def main() -> None:
    """Score the training rows, apply the band policy, report the split, plot it."""
    with open(f"{CHAPTER_DIR}/artifacts/model_meta.json") as f:
        meta = json.load(f)
    features, review_cut = meta["features"], meta["threshold"]
    model = CatBoostClassifier()
    model.load_model(f"{CHAPTER_DIR}/artifacts/model.cbm")

    df = pd.read_csv(f"{CHAPTER_DIR}/data/split/train.csv")
    score = model.predict_proba(df[features])[:, 1]
    outcome = np.array([decide(s, review_cut) for s in score])
    counts = {o: int((outcome == o).sum()) for o in ("approve", "investigate", "block")}
    print(f"review_cut {review_cut:.4f}, block_cut {BLOCK_CUT}: {counts}")

    # score shown on a 0-1000 axis (display only; the decision uses the probability)
    s = (score * 1000).clip(0, 1000)
    lo, hi = review_cut * 1000, BLOCK_CUT * 1000
    bins = np.linspace(0, 1000, 60)
    plt.figure(figsize=(11, 4.2), dpi=150)
    ax = plt.gca()
    for m0, m1, c in [(0, lo, APPROVE), (lo, hi, INVESTIGATE), (hi, 1000, BLOCK)]:
        plt.hist(s[(s >= m0) & (s < m1)], bins=bins, color=c, alpha=0.9)
    for x in (lo, hi):
        plt.axvline(x, color="#555", ls="--", lw=1.2)
    y = plt.ylim()[1] * 0.6
    for x, label, c in [
        (lo / 2, f"approve\n{counts['approve']:,}", APPROVE),
        ((lo + hi) / 2, f"investigate\n{counts['investigate']:,}", INVESTIGATE),
        ((hi + 1000) / 2, f"block\n{counts['block']:,}", BLOCK),
    ]:
        plt.text(x, y, label, ha="center", color=c, fontsize=11, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.yscale("log")
    plt.xlabel("Fraud score (0-1000)")
    plt.ylabel("Transactions (log)")
    plt.title("From a threshold to a policy: score bands and outcomes")
    plt.tight_layout()
    os.makedirs(f"{CHAPTER_DIR}/artifacts", exist_ok=True)
    plt.savefig(f"{CHAPTER_DIR}/artifacts/score_bands.png")
    print("saved artifacts/score_bands.png")


if __name__ == "__main__":
    main()
