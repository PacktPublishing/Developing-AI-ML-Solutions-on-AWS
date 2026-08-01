"""MLflow pyfunc wrapper for the scorecard, used only by the training job.

Kept apart from scorecard_model.py so the inference server never imports mlflow.
Logging the scorecard as a pyfunc model is what lets it be registered in the
MLflow Model Registry and later deployed by SageMaker ModelBuilder straight from
a models:/credit-scorecard/<version> URI — the same path the incumbent and the
challenger both travel, so promotion is a registry decision, not a code change.
"""

import mlflow.pyfunc

from scorecard_model import ScorecardPredictor


class ScorecardPyfunc(mlflow.pyfunc.PythonModel):
    """Load the saved scorecard and return probability of default per row."""

    def load_context(self, context) -> None:
        """Rebuild the predictor from the logged model directory."""
        self._model = ScorecardPredictor.load(context.artifacts["model_dir"])

    def predict(self, context, model_input, params=None):
        """Score a DataFrame (or list of records) of applications."""
        return self._model.predict_proba(model_input)
