"""The 0-1000 fraud score: one implementation, shared by training and serving.

The score is a probability's percentile among legitimate traffic, raised to a
power: score = 1000 * r ** gamma. Keeping the arithmetic here, with numpy as its
only dependency, lets the serving container compute the same score as the
training run without carrying scikit-learn for the fitted transformer.
"""

import numpy as np

# gamma solved from one anchor, 1000 * 0.98 ** gamma = 900: a 2% false-alarm cut
# sits at score 900, where Amazon Fraud Detector's published scale puts a 2% rate.
# gamma = 1 would be the linear score, where FPR = 1 - score/1000.
GAMMA = float(np.log(0.9) / np.log(0.98))

# The two cuts a score is banded by. The block cut is the higher of the two and declines
# the payment; the review cut is the lower and sends the case to an analyst afterwards.
# Both are defaults: training writes the pair it fitted into the model metadata.
REVIEW_CUT = 700  # investigate at and above this score (about a 7% false-alarm rate)
BLOCK_CUT = 900  # the 2% false-alarm anchor the scale was solved from


def to_score(proba, legit: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Score probabilities against a sorted reference of legitimate probabilities."""
    proba = np.asarray(proba, float).ravel()
    r = np.searchsorted(legit, proba, side="right") / len(legit)
    return np.round(1000 * r**gamma).astype(int)


def to_fpr(score, gamma: float = GAMMA) -> np.ndarray:
    """Return the false-positive rate a score cut carries, counting every score above it."""
    return 1 - (np.asarray(score, float) / 1000) ** (1 / gamma)


def decide(
    score: float,
    review_cut: float = REVIEW_CUT,
    block_cut: float = BLOCK_CUT,
) -> str:
    """Band a 0-1000 score into approve, investigate, or block.

    Ordered, first match wins: at or above the block cut the payment is stopped
    at the terminal, at or above the review cut it completes and an analyst picks
    the case up afterwards, and below both it is approved outright.
    """
    if score >= block_cut:
        return "block"
    if score >= review_cut:
        return "investigate"
    return "approve"
