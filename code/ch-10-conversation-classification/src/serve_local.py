# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "boto3"]
# ///
"""Run the model's container as a SageMaker endpoint in local mode.

This exists because starting the container by hand does not test what SageMaker
tests. SageMaker runs `docker run <image> serve`, and that argument replaces
CMD, so an image without a command named serve on PATH exits with
CannotStartContainerError and logs nothing. A plain `docker run <image>` uses
CMD instead and passes happily, which is how that failure reached a deployed
endpoint rather than a laptop.

Local mode starts the container the way the service does, against the same
/invocations and /ping contract, so the startup contract is checked here for the
price of a container start.

The weights are in the image, so there is no model artifact to stage. The image
must be the CPU variant unless this machine has an NVIDIA GPU; `make sm-local`
builds that.

Usage:
  MODEL_IMAGE=ch10-decider:cpu SAGEMAKER_ROLE_ARN=<role> uv run src/serve_local.py
"""

import json
import os
import sys

import categories
from sagemaker.local import LocalSession
from sagemaker.model import Model

IMAGE = os.environ.get("MODEL_IMAGE", "ch10-decider:cpu")
# Local mode never assumes this role; the SDK requires one to construct a Model.
ROLE = os.environ.get("SAGEMAKER_ROLE_ARN", "")
DEVICE = os.environ.get("DEVICE", "cpu")
CONVERSATION = os.environ.get(
    "CONVERSATION", "My debit card was declined at the till this morning."
)


def main() -> None:
    """Deploy the container locally, ask it one question, then take it down."""
    if not ROLE:
        raise SystemExit(
            "set SAGEMAKER_ROLE_ARN; local mode does not assume it, but the SDK "
            "will not build a Model without one"
        )

    session = LocalSession()
    session.config = {"local": {"local_code": True}}

    model = Model(
        image_uri=IMAGE,
        role=ROLE,
        sagemaker_session=session,
        env={"DEVICE": DEVICE},
    )

    print(f"starting {IMAGE} as a local endpoint, device {DEVICE}")
    predictor = model.deploy(initial_instance_count=1, instance_type="local")
    try:
        labels = list(categories.BORDERS)
        request = {
            "state": CONVERSATION,
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this conversation?",
                    "criteria": categories.as_criteria(labels, described=True),
                }
            },
        }
        raw = predictor.predict(
            json.dumps(request).encode(),
            initial_args={"ContentType": "application/json"},
        )
        answer = json.loads(raw)["answers"]["team"]
        print(f"  model:      {json.loads(raw).get('model', 'unnamed')}")
        print(f"  team:       {answer['choice']}")
        print(f"  confidence: {answer['confidence']:.3f}")
    finally:
        # Leaving a local endpoint running holds the port and the weights in
        # memory, so take it down even when the question failed.
        print("deleting the local endpoint")
        predictor.delete_endpoint()


if __name__ == "__main__":
    sys.exit(main())
