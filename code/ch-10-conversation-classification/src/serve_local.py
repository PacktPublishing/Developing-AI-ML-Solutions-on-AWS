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

`make sm-local` builds a CPU image from the deployment Dockerfile with a
different base image and PyTorch build. Both images include the weights. This
checks the serving entrypoint and API, but does not test the deployed CUDA stack.

Usage:
  MODEL_IMAGE=ch10-decider:cpu uv run src/serve_local.py
"""

import importlib
import json
import os
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request

import categories
from sagemaker.core.local import LocalSession

# The local serving path imports sagemaker.serve.model_builder lazily; load it up
# front so the attribute exists when create_endpoint starts the container.
importlib.import_module("sagemaker.serve.model_builder")

IMAGE = os.environ.get("MODEL_IMAGE", "ch10-decider:cpu")
PORT = int(os.environ.get("SERVING_PORT", "8081"))
DEVICE = os.environ.get("DEVICE", "cpu")
# Empty means float32, which is what the endpoint loads: this check is worth
# more when it runs the container the way the service runs it. float32 weights
# need about 8 GB, so Docker Desktop's default VM is too small for them and the
# container is killed with code 137 partway through loading. Raise Docker's
# memory, or set TORCH_DTYPE=bfloat16 to halve the weights and diverge from the
# endpoint deliberately.
DTYPE = os.environ.get("TORCH_DTYPE", "")
NAME = "ch10-decider-local"
CONVERSATION = os.environ.get(
    "CONVERSATION", "My debit card was declined at the till this morning."
)
# Allow time to load and warm the model before the server reports readiness.
TRIES = int(os.environ.get("PING_TRIES", "300"))

# Both images include the weights, so local mode receives an empty model artifact.
_EMPTY_TAR = os.path.join(tempfile.mkdtemp(), "model.tar.gz")
tarfile.open(_EMPTY_TAR, "w:gz").close()


SESSION = LocalSession()
SESSION.config = {"local": {"serving_port": PORT, "local_code": True}}


def serve() -> None:
    """Bring the image up as a local SageMaker endpoint. Blocks."""
    client = SESSION.sagemaker_client
    client.create_model(
        ModelName=NAME,
        PrimaryContainer={
            "Image": IMAGE,
            "ModelDataUrl": f"file://{_EMPTY_TAR}",
            "Environment": {
                "DEVICE": DEVICE,
                "REQUIRE_GPU": "0",
                # Docker Desktop's VM is smaller than this machine, and float32
                # weights do not fit in it. See the note in model/serve.py.
                "TORCH_DTYPE": DTYPE,
            },
        },
    )
    client.create_endpoint_config(
        EndpointConfigName=NAME,
        ProductionVariants=[
            {
                "VariantName": "main",
                "ModelName": NAME,
                "InitialInstanceCount": 1,
                "InstanceType": "local",
            }
        ],
    )
    client.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)


def wait_for_ping() -> bool:
    """Poll the container's /ping until it answers, the way SageMaker does."""
    import time

    for _ in range(TRIES):
        try:
            with urllib.request.urlopen(
                f"http://localhost:{PORT}/ping", timeout=2
            ) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def ask() -> dict:
    """Put one typed question to the container on /invocations."""
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
    body = json.dumps(request).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/invocations",
        data=body,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def main() -> None:
    """Start the container, ask it one question, and report what answered."""
    print(f"starting {IMAGE} as a local endpoint on :{PORT}, device {DEVICE}")
    # create_endpoint stays in the foreground holding the endpoint open; the
    # container itself runs detached.
    threading.Thread(target=serve, daemon=True).start()

    try:
        if not wait_for_ping():
            raise SystemExit(
                f"the container never answered /ping on :{PORT}. "
                "docker ps and docker logs will say why."
            )
        print("  /ping answered, so the container started the server we asked for")

        answer = ask()["answers"]["team"]
        print(f"  team:       {answer['choice']}")
        print(f"  confidence: {answer['confidence']:.3f}")
    finally:
        # create_endpoint runs in a daemon thread, so this process exiting does
        # not stop the container: it keeps the port and several GB of weights
        # until something takes it down. Delete the endpoint even when the
        # question failed, which is most of the times this script is run.
        print("deleting the local endpoint")
        SESSION.sagemaker_client.delete_endpoint(EndpointName=NAME)


if __name__ == "__main__":
    main()
