# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "boto3", "fastapi", "uvicorn", "httpx", "pydantic>=2"]
# ///
"""Run the decision gateway over two models served as concurrent SageMaker LOCAL endpoints.

SageMaker local mode serves each model in its own container. The SDK pins the serving port at
8080, so two models would collide; local.serving_port moves each one, and scorecard and
challenger run side by side on 8081 and 8082. The gateway then ensembles them exactly as it
ensembles two SageMaker endpoints on AWS. Each endpoint is brought up in a daemon thread
because create_endpoint stays in the foreground holding the endpoint open; the containers run
detached. Needs the two model images built and DynamoDB reachable (make up starts it).
"""

import importlib
import os
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request

from sagemaker.core.local import LocalSession

# the local serving path imports sagemaker.serve.model_builder lazily; load it up front so the
# attribute exists when create_endpoint starts the container
importlib.import_module("sagemaker.serve.model_builder")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"scorecard": 8081, "challenger": 8082}

# the model is baked into the BYOC image, so the model data is an empty tar
_EMPTY_TAR = os.path.join(tempfile.mkdtemp(), "model.tar.gz")
tarfile.open(_EMPTY_TAR, "w:gz").close()


def _serve(model: str, port: int) -> None:
    """Bring up one model as a local SageMaker endpoint on its own port (blocks)."""
    sess = LocalSession()
    sess.config = {"local": {"serving_port": port, "local_code": True}}
    client = sess.sagemaker_client
    client.create_model(
        ModelName=model,
        PrimaryContainer={
            "Image": f"ch04-{model}-model:latest",
            "ModelDataUrl": f"file://{_EMPTY_TAR}",
            "Environment": {},
        },
    )
    client.create_endpoint_config(
        EndpointConfigName=model,
        ProductionVariants=[
            {
                "VariantName": "A",
                "ModelName": model,
                "InitialInstanceCount": 1,
                "InstanceType": "local",
            }
        ],
    )
    client.create_endpoint(EndpointName=model, EndpointConfigName=model)


def _wait(port: int, tries: int = 90) -> None:
    """Poll a container's /ping until it answers."""
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/ping", timeout=2)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise RuntimeError(f"model on :{port} did not come up")


def main() -> None:
    """Serve both models, then run the gateway over them."""
    for model, port in MODELS.items():
        threading.Thread(target=_serve, args=(model, port), daemon=True).start()
    for port in MODELS.values():
        _wait(port)
    print("model endpoints up:", MODELS, flush=True)

    # point the gateway at the two local endpoints and run it (its code is unchanged: a http
    # target is scored over the wire, the same _invoke path a SageMaker endpoint takes on AWS)
    os.environ["ENDPOINTS"] = ",".join(
        f"{m}=http://localhost:{p}" for m, p in MODELS.items()
    )
    os.environ.setdefault("DECISIONS_TABLE", "decision_log")
    os.environ.setdefault("DYNAMODB_ENDPOINT", "http://localhost:8000")
    sys.path.insert(0, os.path.join(ROOT, "src", "gateway"))
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
