"""SageMaker inference server for the KYC face service (custom container).

Serves the SageMaker contract on :8080 (GET /ping, POST /invocations) and does no
inference of its own: it hands the request to the same FaceEmbeddingSpec that
ModelBuilder serves in process on a laptop. The container is the transport; the spec
is the model. Behind an asynchronous endpoint SageMaker fetches the request from S3
and posts it here unchanged, so this file never learns which kind of endpoint it is.
"""

import json
import logging
import os

import torch
from face_spec import FaceEmbeddingSpec
from flask import Flask, Response, request

log = logging.getLogger("serve")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Loaded once per container, exactly as the in-process server loads it once per run.
# Log what torch can actually see, so a container that silently lands on the CPU
# on a GPU instance says so in the endpoint's first log line rather than in a
# "device": "cpu" buried in a result object.
log.info(
    "torch %s | cuda build %s | cuda available %s | devices %d",
    torch.__version__,
    torch.version.cuda,
    torch.cuda.is_available(),
    torch.cuda.device_count(),
)
if not torch.cuda.is_available():
    # device_count can be non-zero while is_available is False: the device is visible
    # but the runtime cannot initialise it, usually a driver older than the CUDA build.
    try:
        torch.cuda.init()
    except (RuntimeError, AssertionError, OSError) as exc:
        log.error("cuda init failed: %s: %s", type(exc).__name__, exc)

# An endpoint paid for by the GPU-hour that quietly runs on the CPU is the failure
# this chapter already made once: it answers correctly, so nothing looks wrong. Fail
# the health check instead, unless the deploy explicitly asked for the CPU.
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "1") == "1"
GPU_MISSING = REQUIRE_GPU and not torch.cuda.is_available()
if GPU_MISSING:
    log.error(
        "REQUIRE_GPU is set but torch reports no CUDA device (build %s). "
        "/ping will fail so the endpoint does not come InService on the CPU.",
        torch.version.cuda,
    )

_spec = FaceEmbeddingSpec()
_model = _spec.load()


@app.route("/ping", methods=["GET"])
def ping() -> Response:
    """Health check: ready once the model is loaded on the device we asked for."""
    if GPU_MISSING:
        return Response("no CUDA device", status=503)
    return Response(status=200 if _model is not None else 503)


@app.route("/invocations", methods=["POST"])
@app.route("/endpoints/<name>/invocations", methods=["POST"])
def invocations(name: str | None = None) -> Response:
    """Answer whichever operation the body asks for."""
    params = json.loads(request.get_data() or b"{}")
    result = _spec.invoke(params, _model)
    status = 400 if "error" in result else 200
    log.info(
        "%s -> %s",
        params.get("op", "match"),
        {k: v for k, v in result.items() if k != "matches"},
    )
    return Response(json.dumps(result), status=status, mimetype="application/json")
