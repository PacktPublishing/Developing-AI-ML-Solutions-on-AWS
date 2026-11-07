# /// script
# dependencies = ["bedrock-agentcore", "strands-decider"]
# ///
"""Serve the decision model as an AgentCore Runtime agent.

BedrockAgentCoreApp is the runtime contract: it serves POST /invocations and
GET /ping, which is what AgentCore invokes a container on. The same file runs on
a laptop and in the cloud, so the local round is the deployed path rather than a
stand-in for it.

The payload is a System One request, the shape the Strands Decider and Kev
servers take, so src/decider.py reaches this app with DECIDER_BACKEND=http and
no client change:

  {"state": "<the conversation>",
   "questions": {"team": {"type": "choice",
                          "instructions": "Which team should handle this?",
                          "criteria": {"Cards": "...", "Mortgages": null}}}}

There is one other payload, {"mode": "prepare"}, and it exists because of a
limit rather than a feature. Neither artifact AgentCore accepts can carry this
model: a container image stops at 2 GB where CUDA torch alone is 4.2 GB, and an
S3 source package at 750 MB unpacked where the weights are 4.8 GB. So the image
holds the agent and nothing else, and the first call of a session installs torch
and fetches the weights onto the mounted volume. The volume survives session
stops, so only the first call pays for it.

Usage:
  uv run agentcore/app.py                  # local, http://127.0.0.1:8080
"""

import os
import subprocess
import sys
import time

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Kept here rather than imported from src/ so the deployable unit is this
# directory alone. It matches the default in src/decider.py.
DECIDER_MODEL = os.environ.get(
    "DECIDER_MODEL", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
)
DECIDER_DEVICE = os.environ.get("DECIDER_DEVICE", "")
DECIDER_VERSION = os.environ.get("DECIDER_VERSION", "0.3.0")
# Where the capacity provider mounts its EBS volume. Empty locally, where the
# dependencies are already in the environment and there is nothing to prepare.
MODEL_VOLUME = os.environ.get("MODEL_VOLUME", "")
# The CUDA build, because this runs on a GPU instance. It is also why torch
# cannot be in the image.
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu124")

VOLUME_PACKAGES = os.path.join(MODEL_VOLUME, "env") if MODEL_VOLUME else ""

app = BedrockAgentCoreApp()
_engine = None


def _use_volume_packages() -> None:
    """Import from the volume as well as from the image."""
    if VOLUME_PACKAGES and VOLUME_PACKAGES not in sys.path:
        sys.path.insert(0, VOLUME_PACKAGES)


def _device() -> str:
    """Return the device to load on: the override, else the best one present."""
    if DECIDER_DEVICE:
        return DECIDER_DEVICE
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare() -> dict:
    """Install torch and the model onto the volume, and report what it cost.

    Runs once per volume, not once per invocation: a session that stops and
    resumes re-attaches the same volume with this already done.
    """
    if not MODEL_VOLUME:
        return {
            "error": "MODEL_VOLUME is not set, so there is no volume to prepare. "
            "Locally the dependencies are already installed."
        }
    started = time.perf_counter()
    os.makedirs(VOLUME_PACKAGES, exist_ok=True)
    # A fixed argument list and no shell: nothing here comes from the payload.
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            VOLUME_PACKAGES,
            "--index-url",
            TORCH_INDEX,
            "torch",
        ],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            VOLUME_PACKAGES,
            f"strands-decider=={DECIDER_VERSION}",
        ],
        check=True,
    )
    installed = time.perf_counter() - started
    # Loading downloads the checkpoint into HF_HOME, which is on the volume too,
    # so the weights are fetched here rather than on the first real question.
    engine()
    return {
        "prepared": True,
        "volume": MODEL_VOLUME,
        "install_seconds": round(installed, 1),
        "total_seconds": round(time.perf_counter() - started, 1),
        "device": _device(),
    }


def engine():
    """Load the checkpoint once per container and reuse it."""
    global _engine
    if _engine is None:
        _use_volume_packages()
        try:
            from strands_decider.infer import load_engine
        except ImportError as err:
            raise RuntimeError(
                "the decision model is not installed; invoke this runtime once "
                'with {"mode": "prepare"} to put it on the volume'
            ) from err
        _engine = load_engine(DECIDER_MODEL, device=_device())
    return _engine


@app.entrypoint
def decide(payload: dict) -> dict:
    """Answer the typed questions in one System One request, or prepare the volume.

    The SDK hands the payload over unchanged, so it is checked here. A missing
    or malformed request is answered with the reason rather than a stack trace,
    because the caller is a batch loop over someone's conversations and one bad
    request should not read like the model failing.
    """
    if not isinstance(payload, dict):
        return {"error": "the payload must be a JSON object"}
    if payload.get("mode") == "prepare":
        return prepare()
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        return {"error": "questions must be a non-empty object of typed questions"}
    state = payload.get("state", "")
    if not isinstance(state, str):
        return {"error": "state must be a string"}
    try:
        answered = engine().ask(state, questions).model_dump()
    except RuntimeError as err:
        return {"error": str(err)}
    # The engine names itself after the strands-decider package version, which
    # says nothing about which model answered. Report the checkpoint, so a
    # predictions file records what produced it.
    answered["model"] = DECIDER_MODEL
    return answered


if __name__ == "__main__":
    # 8080 is the port AgentCore invokes; the SDK binds 127.0.0.1 here and
    # 0.0.0.0 inside a container, so the same call works in both places.
    app.run(port=int(os.environ.get("PORT", "8080")))
