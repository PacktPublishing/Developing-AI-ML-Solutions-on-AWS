"""The XGBoost challenger, shared by training and serving.

The data scientist's challenge to the incumbent WOE scorecard: gradient-boosted
trees, which usually score higher but do not honor the bank's monotonic business
rule on their own. So the same rule is imposed here through XGBoost's
monotone_constraints — a higher bureau score can only lower predicted risk, a
higher debt-to-income can only raise it. Categoricals carry no direction and are
handled by XGBoost's native categorical support (constraint 0).

Pure xgboost/pandas (no mlflow, no web framework) so both the training job and
the inference server import it, and so it packages cleanly into an MLflow pyfunc.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import xgboost as xgb

ARTIFACT = "challenger.ubj"
SPEC_FILE = "feature_spec.json"


class ChallengerModel:
    """A fitted monotone XGBoost classifier with a single scoring call."""

    def __init__(self, model: xgb.XGBClassifier, spec: dict) -> None:
        """Hold the fitted XGBoost model and the feature spec."""
        self.model = model
        self.spec = spec
        self.numeric = spec["numeric_features"]
        self.categorical = spec["categorical_features"]
        self.features = self.numeric + self.categorical

    def _frame(self, rows) -> pd.DataFrame:
        """Coerce input into the DataFrame shape XGBoost trained on."""
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        df = df[self.features].copy()
        for col in self.categorical:
            df[col] = df[col].astype("category")
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
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(os.path.join(model_dir, ARTIFACT))
        with open(os.path.join(model_dir, SPEC_FILE)) as fh:
            spec = json.load(fh)
        return cls(model, spec)
