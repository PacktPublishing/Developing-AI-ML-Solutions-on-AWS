"""The scorecard model, shared by training and serving.

A Weight-of-Evidence logistic-regression scorecard: FastWoe encodes every
feature to its WOE (numerics are monotone tree-binned, categoricals target-
encoded), then a scikit-learn LogisticRegression scores the WOE vector. The
monotone binning is where the bank's business rule lives — a higher bureau
score can only lower predicted risk — so the incumbent model satisfies it by
construction, without a constraint being bolted on afterward.

This module is pure numpy/sklearn/fastwoe (no mlflow, no web framework) so both
the training job and the inference server can import it, and so it packages
cleanly into an MLflow pyfunc via code_paths.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd

ARTIFACT = "scorecard.joblib"
SPEC_FILE = "feature_spec.json"


class ScorecardPredictor:
    """A fitted WOE + logistic-regression scorecard with a single scoring call."""

    def __init__(self, woe, lr, spec: dict) -> None:
        """Hold the fitted WOE encoder, the logistic regression, and the feature spec."""
        self.woe = woe
        self.lr = lr
        self.spec = spec
        self.features = spec["numeric_features"] + spec["categorical_features"]

    def predict_proba(self, rows) -> list[float]:
        """Return the probability of default for each input row.

        Accepts a DataFrame or a list of dicts with the model's feature columns.
        """
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        df = df[self.features]
        woe = self.woe.transform(df)
        return self.lr.predict_proba(woe)[:, 1].tolist()

    def save(self, model_dir: str) -> None:
        """Persist the model to a SageMaker model directory."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(
            {"woe": self.woe, "lr": self.lr, "spec": self.spec},
            os.path.join(model_dir, ARTIFACT),
        )
        with open(os.path.join(model_dir, SPEC_FILE), "w") as fh:
            json.dump(self.spec, fh, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "ScorecardPredictor":
        """Load a model previously written by save()."""
        blob = joblib.load(os.path.join(model_dir, ARTIFACT))
        return cls(blob["woe"], blob["lr"], blob["spec"])
