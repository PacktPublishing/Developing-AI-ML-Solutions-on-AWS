"""The SageMaker training entrypoint: fit the scorecard against the path contract.

SageMaker mounts the same paths whether the job runs in local mode (a container on
this machine) or as a managed job on AWS, so the one script trains both:

  /opt/ml/input/data/reference/    reference.csv (the channel the batch is trained on)
  /opt/ml/input/config/hyperparameters.json
  /opt/ml/model/                   the fitted scorecard is written here
  /opt/ml/output/failure           a readable reason if training fails

Local mode and the managed job differ only in where those paths are backed -- a
bind mount here, an S3 download there. The image is identical.
"""

import json
import os
import traceback

import pandas as pd
from model import FEATURES, train

PREFIX = "/opt/ml"
REFERENCE = f"{PREFIX}/input/data/reference"
CONFIG = f"{PREFIX}/input/config/hyperparameters.json"
MODEL = f"{PREFIX}/model"
FAILURE = f"{PREFIX}/output/failure"


def _hyperparameters() -> dict:
    """Read SageMaker hyperparameters (every value arrives as a string)."""
    if os.path.exists(CONFIG):
        with open(CONFIG) as fh:
            return json.load(fh)
    return {}


def main() -> None:
    """Train the scorecard on the reference channel and write it to the model dir."""
    try:
        reference = pd.read_csv(os.path.join(REFERENCE, "reference.csv"))
        model = train(reference, _hyperparameters())

        os.makedirs(MODEL, exist_ok=True)
        model.save_model(os.path.join(MODEL, "scorecard.cbm"))

        top = sorted(zip(FEATURES, model.feature_importances_), key=lambda kv: -kv[1])[
            :5
        ]
        print("trained scorecard; top features:", [f"{n} {i:.1f}" for n, i in top])
        print(f"saved -> {MODEL}/scorecard.cbm")
    except Exception:
        os.makedirs(os.path.dirname(FAILURE), exist_ok=True)
        with open(FAILURE, "w") as fh:
            fh.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
