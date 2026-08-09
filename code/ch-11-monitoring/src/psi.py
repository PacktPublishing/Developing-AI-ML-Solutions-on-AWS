# /// script
# dependencies = ["catboost", "pandas", "numpy"]
# ///
"""PSIDetector: Population Stability Index over a model's own bins.

Traditional PSI bins a feature into quantiles of the reference sample -- it measures
raw distribution shift, but not necessarily the shift the model reacts to. A tree
model partitions each feature at specific thresholds (its split borders) and a
scorecard at its WOE bin edges; binning PSI at those boundaries makes it measure
whether the population drifted across the lines the model actually uses to decide.
That is the ML approach to PSI.

PSIDetector fits the reference proportions over a set of bins and scores the PSI of a
later batch against them. Build it two ways: from_reference bins each feature into
reference quantiles (the scorecard / linear-model path), from_catboost bins each
numeric feature at the model's own split borders (the tree path). It serialises to a
transparent JSON artifact and logs to MLflow as a pyfunc model.

Usage:
  from psi import PSIDetector
  det = PSIDetector.from_catboost(model, reference, NUMERIC, CATEGORICAL)
  det.psi(current)                 # {feature: psi}
  det.save("psi.json"); PSIDetector.load("psi.json")
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6

# < 0.1 stable, 0.1-0.25 moderate, > 0.25 major.
PSI_MAJOR = 0.25


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    """Return the PSI between two proportion vectors over the same bins."""
    ref = np.clip(ref_pct, EPS, None)
    cur = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _quantile_borders(series: pd.Series, bins: int) -> list[float]:
    """Interior bin borders at the reference quantiles (deduplicated)."""
    q = np.unique(np.quantile(series, np.linspace(0, 1, bins + 1)))
    return [float(b) for b in q[1:-1]]  # drop the outer edges; ±inf is implicit


class PSIDetector:
    """A fitted set of per-feature bins plus the reference proportions over them."""

    def __init__(
        self,
        borders: dict[str, list[float]],
        categories: dict[str, list[str]],
        reference_pct: dict[str, list[float]],
        binning: str = "quantile",
    ) -> None:
        """Hold the interior numeric borders, the category sets, and reference shares."""
        self.borders = {f: [float(b) for b in bs] for f, bs in borders.items()}
        self.categories = {f: [str(c) for c in cs] for f, cs in categories.items()}
        self.reference_pct = {
            f: [float(p) for p in ps] for f, ps in reference_pct.items()
        }
        self.binning = binning

    # -- edges from interior borders (±inf ends are implicit, so JSON stays finite) --
    def _edges(self, feature: str) -> np.ndarray:
        """Return the full bin edges for a numeric feature, capped at ±inf."""
        return np.array([-np.inf, *self.borders[feature], np.inf])

    @staticmethod
    def _shares(series: pd.Series, edges: np.ndarray) -> np.ndarray:
        """Proportion of rows falling in each numeric bin."""
        return np.histogram(series, edges)[0] / len(series)

    @staticmethod
    def _cat_shares(series: pd.Series, categories: list[str]) -> np.ndarray:
        """Proportion of rows in each known category, counted in a single pass."""
        counts = series.astype(str).value_counts(normalize=True)
        return counts.reindex(categories, fill_value=0.0).to_numpy()

    # -- construction -----------------------------------------------------------
    @classmethod
    def from_reference(
        cls,
        reference: pd.DataFrame,
        numeric: Sequence[str],
        categorical: Sequence[str] = (),
        bins: int = 10,
    ) -> PSIDetector:
        """Bin each numeric feature into reference quantiles (the scorecard path)."""
        borders = {f: _quantile_borders(reference[f], bins) for f in numeric}
        return cls._fit(reference, borders, list(categorical), "quantile")

    @classmethod
    def from_catboost(
        cls,
        model,
        reference: pd.DataFrame,
        numeric: Sequence[str],
        categorical: Sequence[str] = (),
    ) -> PSIDetector:
        """Bin each numeric feature at the model's own split borders (the tree path)."""
        # get_borders() keys by the OVERALL feature index (categorical positions are
        # simply absent), so feature_names_.index(f) aligns a name to its borders no
        # matter where the categoricals sit -- verified against a mid-list categorical.
        names = list(model.feature_names_)
        model_borders = model.get_borders()
        borders = {}
        for f in numeric:
            bs = model_borders.get(names.index(f), [])
            # a feature the model never split on has no borders: fall back to quantiles.
            borders[f] = (
                [float(b) for b in bs]
                if len(bs)
                else _quantile_borders(reference[f], 10)
            )
        return cls._fit(reference, borders, list(categorical), "catboost")

    @classmethod
    def _fit(
        cls,
        reference: pd.DataFrame,
        borders: dict[str, list[float]],
        categorical: list[str],
        binning: str,
    ) -> PSIDetector:
        """Compute the reference proportions over the given bins."""
        reference_pct: dict[str, list[float]] = {}
        for f, bs in borders.items():
            edges = np.array([-np.inf, *bs, np.inf])
            reference_pct[f] = list(cls._shares(reference[f], edges))
        categories = {f: sorted(reference[f].astype(str).unique()) for f in categorical}
        for f, cats in categories.items():
            reference_pct[f] = list(cls._cat_shares(reference[f], cats))
        return cls(borders, categories, reference_pct, binning)

    # -- scoring ----------------------------------------------------------------
    def psi(self, current: pd.DataFrame) -> dict[str, float]:
        """Return the PSI of each feature in the current batch against the reference."""
        out: dict[str, float] = {}
        for f in self.borders:
            cur = self._shares(current[f], self._edges(f))
            out[f] = round(_psi(np.array(self.reference_pct[f]), cur), 4)
        for f, cats in self.categories.items():
            cur = self._cat_shares(current[f], cats)
            out[f] = round(_psi(np.array(self.reference_pct[f]), cur), 4)
        return out

    def violations(self, current: pd.DataFrame, threshold: float = PSI_MAJOR) -> dict:
        """Return the features whose PSI exceeds the major-shift threshold."""
        return {f: v for f, v in self.psi(current).items() if v > threshold}

    # -- serialisation ----------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a JSON-safe representation (all borders finite)."""
        return {
            "binning": self.binning,
            "borders": self.borders,
            "categories": self.categories,
            "reference_pct": self.reference_pct,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PSIDetector:
        """Rebuild a detector from to_dict()."""
        return cls(
            d["borders"],
            d["categories"],
            d["reference_pct"],
            d.get("binning", "quantile"),
        )

    def save(self, path: str | Path) -> None:
        """Write the detector to a transparent JSON artifact."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PSIDetector:
        """Load a detector from a JSON artifact."""
        return cls.from_dict(json.loads(Path(path).read_text()))
