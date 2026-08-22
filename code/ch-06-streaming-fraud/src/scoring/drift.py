# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "catboost", "scikit-learn"]
# ///
"""Score-drift attribution as a fitted, serializable estimator (the DBShap piece).

ScoreDriftAttributor splits the mean-score drift between a reference and a live batch across
the input features. The value function is label-free: for a coalition of features, draw those
features from the live marginal and the rest from the reference marginal, score, and take the
mean-score shift. Exact Shapley then gives each feature fair credit, and the credits sum to the
total drift. This is the virtual-drift (P(x)) side of Edakunni et al., "Explaining Drift using
Shapley Values". A fitted instance holds the model, the calibrator, and the reference marginals,
so it pickles and can be reused batch to batch.

Usage:
  attributor = ScoreDriftAttributor(model, calibrator, features).fit(reference)
  contributions = attributor.attribute(live)   # a signed pandas Series, summing to the drift
"""

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ScoreDriftAttributor(BaseEstimator):
    """Attribute the mean-score drift from a reference to a live batch across input features."""

    def __init__(
        self, model, calibrator, features, n_samples: int = 2000, random_state: int = 6
    ):
        self.model = model
        self.calibrator = calibrator
        self.features = list(features)
        self.n_samples = n_samples
        self.random_state = random_state

    def _score(self, frame: pd.DataFrame) -> np.ndarray:
        """Score a feature frame: model probability through the fitted calibrator."""
        proba = self.model.predict_proba(frame[self.features])[:, 1]
        return self.calibrator.transform(proba).ravel()

    def fit(self, reference: pd.DataFrame) -> "ScoreDriftAttributor":
        """Freeze the reference feature marginals the drift is measured against."""
        self.reference_ = reference[self.features].reset_index(drop=True)
        return self

    def attribute(self, live: pd.DataFrame) -> pd.Series:
        """Return each feature's Shapley contribution to the mean-score drift (signed, sums to total)."""
        rng = np.random.default_rng(self.random_state)
        n, N = len(self.features), self.n_samples
        ref_mat = np.column_stack(
            [rng.choice(self.reference_[f].to_numpy(), N) for f in self.features]
        )
        live_mat = np.column_stack(
            [rng.choice(live[f].to_numpy(), N) for f in self.features]
        )

        # value of every coalition: mean-score shift when its features come from the live marginal
        big = np.empty((2**n * N, n))
        for coalition in range(2**n):
            bits = np.array([(coalition >> j) & 1 for j in range(n)], dtype=bool)
            big[coalition * N : (coalition + 1) * N] = np.where(
                bits[None, :], live_mat, ref_mat
            )
        value = (
            self._score(pd.DataFrame(big, columns=self.features))
            .reshape(2**n, N)
            .mean(axis=1)
        )
        value = value - value[0]  # v(empty) = 0

        weight = {
            k: factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)
        }
        phi = np.zeros(n)
        for j in range(n):
            others = [i for i in range(n) if i != j]
            for k in range(len(others) + 1):
                for subset in combinations(others, k):
                    s = sum(1 << i for i in subset)
                    phi[j] += weight[k] * (value[s | (1 << j)] - value[s])
        return pd.Series(phi, index=self.features).sort_values(
            key=np.abs, ascending=False
        )
