"""The CatBoost challenger, shared by training and serving.

Gradient-boosted trees with native categorical handling: the bank's directional business
rule is imposed through CatBoost's monotone_constraints on the numeric features, and the
categoricals are encoded natively, so an unseen category value at serving time is handled
rather than a hard error. Pure catboost/pandas (no mlflow, no web framework) so training
and serving both import it and it packages cleanly into an MLflow pyfunc.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from catboost import CatBoostClassifier

ARTIFACT = "challenger.cbm"
SPEC_FILE = "feature_spec.json"


class ChallengerModel:
    """A fitted monotone CatBoost classifier with a single scoring call."""

    def __init__(self, model: CatBoostClassifier, spec: dict) -> None:
        """Hold the fitted CatBoost model and the feature spec."""
        self.model = model
        self.spec = spec
        self.numeric = spec["numeric_features"]
        self.categorical = spec["categorical_features"]
        self.features = self.numeric + self.categorical

    def _frame(self, rows) -> pd.DataFrame:
        """Coerce input into the DataFrame shape CatBoost trained on (categoricals as str)."""
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        df = df[self.features].copy()
        for col in self.categorical:
            df[col] = df[col].astype(str)
        return df

    def predict_proba(self, rows) -> list[float]:
        """Return the probability of default for each input row."""
        return self.model.predict_proba(self._frame(rows))[:, 1].tolist()

    def save(self, model_dir: str) -> None:
        """Persist the model to a SageMaker model directory."""
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_model(os.path.join(model_dir, ARTIFACT))
        with open(os.path.join(model_dir, SPEC_FILE), "w") as fh:
            json.dump(self.spec, fh, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "ChallengerModel":
        """Load a model previously written by save()."""
        model = CatBoostClassifier()
        model.load_model(os.path.join(model_dir, ARTIFACT))
        with open(os.path.join(model_dir, SPEC_FILE)) as fh:
            spec = json.load(fh)
        return cls(model, spec)
