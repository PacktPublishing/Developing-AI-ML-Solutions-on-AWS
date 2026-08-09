# /// script
# dependencies = ["mlflow", "catboost", "pandas", "numpy"]
# ///
"""Log a PSIDetector to MLflow as a pyfunc model, and load it back.

The detector serialises to a single JSON artifact (PSIDetector.save); this wraps that
artifact as an MLflow pyfunc model so the drift baseline carries the same lineage,
versioning, and registry as the model it watches. predict() takes a batch and returns
one row of per-feature PSI. Kept apart from psi.py so the detector itself needs no
MLflow.

Usage:
  import mlflow
  from psi import PSIDetector
  from mlflow_psi import log_detector
  det = PSIDetector.from_catboost(model, reference, NUMERIC, CATEGORICAL)
  with mlflow.start_run():
      log_detector(det)                         # -> registered pyfunc model
  loaded = mlflow.pyfunc.load_model(model_uri)  # loaded.predict(current) -> PSI row
"""

import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from psi import PSIDetector

ARTIFACT_KEY = "psi_detector"


class PSIModel(mlflow.pyfunc.PythonModel):
    """A pyfunc that scores a batch's PSI against a saved PSIDetector."""

    def load_context(self, context) -> None:
        """Load the detector from its JSON artifact."""
        self.detector = PSIDetector.load(context.artifacts[ARTIFACT_KEY])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """Return one row of per-feature PSI for the input batch."""
        return pd.DataFrame([self.detector.psi(model_input)])


def log_detector(detector: PSIDetector, artifact_path: str = "psi_detector"):
    """Log a fitted detector as a pyfunc model under the active MLflow run."""
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "psi.json"
        detector.save(artifact)
        return mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=PSIModel(),
            artifacts={ARTIFACT_KEY: str(artifact)},
        )
