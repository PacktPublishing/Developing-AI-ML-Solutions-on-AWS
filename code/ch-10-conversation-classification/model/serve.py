"""SageMaker inference server for the decision model (custom container).

Serves the SageMaker contract on :8080 (GET /ping, POST /invocations) and does
no inference of its own: strands-decider's own server already implements that
contract, including the batching and the warmup, so this file supplies what the
package cannot know about the host it landed on.

That is the whole reason it exists. A container whose torch was built for a
newer CUDA than the host driver supports dies inside `torch._C._cuda_init()`
with a stack trace, several screens down, after the endpoint has already spent
twenty minutes provisioning. The first lines of this log say what torch is, what
it was built against, and whether the driver can actually start it, so the
answer is at the top of the log rather than buried in a traceback.

Usage:
  CHECKPOINT=<hf-id> DEVICE=cuda python serve.py
"""

import logging
import os
import sys

import torch

log = logging.getLogger("serve")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

CHECKPOINT = os.environ.get(
    "CHECKPOINT", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
)
DEVICE = os.environ.get("DEVICE", "cuda")
PORT = int(os.environ.get("PORT", "8080"))
# An endpoint paid for by the GPU-hour that quietly runs on the CPU answers
# correctly, so nothing looks wrong until the bill does. Refuse to start instead,
# unless the deploy asked for the CPU on purpose.
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "1") == "1" and DEVICE != "cpu"

# Log what torch can actually see before anything tries to use it. device_count
# can be non-zero while is_available is False: the device is visible but the
# runtime cannot initialize it, which is what an out-of-date driver looks like.
log.info(
    "torch %s | cuda build %s | cuda available %s | devices %d",
    torch.__version__,
    torch.version.cuda,
    torch.cuda.is_available(),
    torch.cuda.device_count(),
)
if DEVICE == "cuda" and not torch.cuda.is_available():
    try:
        torch.cuda.init()
    except (RuntimeError, AssertionError, OSError) as exc:
        log.error("cuda init failed: %s: %s", type(exc).__name__, exc)

if REQUIRE_GPU and not torch.cuda.is_available():
    log.error(
        "REQUIRE_GPU is set but torch reports no usable CUDA device (torch was "
        "built against CUDA %s). Exiting rather than coming InService on the "
        "CPU. Set DEVICE=cpu to run here deliberately.",
        torch.version.cuda,
    )
    sys.exit(1)


def main() -> None:
    """Serve the decision model on the SageMaker contract."""
    import uvicorn
    from strands_decider.server import create_app

    log.info("loading %s on %s", CHECKPOINT, DEVICE)
    # create_app warms the model, so uvicorn only starts serving once the
    # weights are resident. That is what lets /ping mean ready rather than
    # listening, which is the difference SageMaker's health check cares about.
    app = create_app(CHECKPOINT, device=DEVICE)
    log.info("serving on 0.0.0.0:%d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
