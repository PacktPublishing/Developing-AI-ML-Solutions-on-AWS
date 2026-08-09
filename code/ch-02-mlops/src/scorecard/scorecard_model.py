"""The scorecard model: a scikit-learn WOE + logistic-regression pipeline.

FastWoe is a scikit-learn transformer (numerics monotone tree-binned, categoricals
target-encoded), so the scorecard is a Pipeline([woe, lr]) and the monotone binning
carries the bank's business rule by construction. The pipeline serialises natively:
joblib for the serving container, mlflow.sklearn for the registry. ScorecardPredictor is
a thin wrapper that selects the model's features and returns the probability of default.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd

ARTIFACT = "scorecard.joblib"
SPEC_FILE = "feature_spec.json"


class ScorecardPredictor:
    """A fitted WOE + logistic-regression pipeline with a single scoring call."""

    def __init__(self, pipeline, spec: dict) -> None:
        """Hold the fitted scikit-learn pipeline and the feature spec."""
        self.pipeline = pipeline
        self.spec = spec
        self.features = spec["numeric_features"] + spec["categorical_features"]

    def predict_proba(self, rows) -> list[float]:
        """Return the probability of default for each input row.

        Accepts a DataFrame or a list of dicts with the model's feature columns.
        """
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        return self.pipeline.predict_proba(df[self.features])[:, 1].tolist()

    def save(self, model_dir: str) -> None:
        """Persist the pipeline to a SageMaker model directory."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "spec": self.spec},
            os.path.join(model_dir, ARTIFACT),
        )
        with open(os.path.join(model_dir, SPEC_FILE), "w") as fh:
            json.dump(self.spec, fh, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "ScorecardPredictor":
        """Load a pipeline previously written by save()."""
        blob = joblib.load(os.path.join(model_dir, ARTIFACT))
        return cls(blob["pipeline"], blob["spec"])
