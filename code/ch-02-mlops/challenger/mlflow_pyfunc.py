"""MLflow pyfunc wrapper for the challenger, used only by the training job.

Kept apart from challenger_model.py so the inference server never imports mlflow.
Logging the challenger as a pyfunc model registers it alongside the incumbent, so
a promotion decision — keep the scorecard or switch to the challenger — is a
registry action rather than a code change.
"""

import mlflow.pyfunc

from challenger_model import ChallengerModel


class ChallengerPyfunc(mlflow.pyfunc.PythonModel):
    """Load the saved challenger and return probability of default per row."""

    def load_context(self, context) -> None:
        """Rebuild the model from the logged model directory."""
        self._model = ChallengerModel.load(context.artifacts["model_dir"])

    def predict(self, context, model_input, params=None):
        """Score a DataFrame (or list of records) of applications."""
        return self._model.predict_proba(model_input)
