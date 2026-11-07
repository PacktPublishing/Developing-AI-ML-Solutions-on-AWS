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

Usage:
  uv run agentcore/app.py                  # local, http://127.0.0.1:8080
  agentcore configure -e agentcore/app.py  # then agentcore launch
"""

import os

from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Kept here rather than imported from src/ so the deployable unit is this
# directory alone. It matches the default in src/decider.py.
DECIDER_MODEL = os.environ.get(
    "DECIDER_MODEL", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
)
DECIDER_DEVICE = os.environ.get("DECIDER_DEVICE", "")

app = BedrockAgentCoreApp()
_engine = None


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


def engine():
    """Load the checkpoint once per container and reuse it.

    The weights are the expensive part, tens of seconds, so the first
    invocation of a session pays for them and the rest of the session does not.
    That is why the router holds one session id for a whole run.
    """
    global _engine
    if _engine is None:
        from strands_decider.infer import load_engine

        _engine = load_engine(DECIDER_MODEL, device=_device())
    return _engine


@app.entrypoint
def decide(payload: dict) -> dict:
    """Answer the typed questions in one System One request.

    The SDK hands the payload over unchanged, so it is checked here. A missing
    or malformed request is answered with the reason rather than a stack trace,
    because the caller is a batch loop over someone's conversations and one bad
    request should not read like the model failing.
    """
    if not isinstance(payload, dict):
        return {"error": "the payload must be a JSON object"}
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        return {"error": "questions must be a non-empty object of typed questions"}
    state = payload.get("state", "")
    if not isinstance(state, str):
        return {"error": "state must be a string"}
    answered = engine().ask(state, questions).model_dump()
    # The engine names itself after the strands-decider package version, which
    # says nothing about which model answered. Report the checkpoint, so a
    # predictions file records what produced it.
    answered["model"] = DECIDER_MODEL
    return answered


if __name__ == "__main__":
    # 8080 is the port AgentCore invokes; the SDK binds 127.0.0.1 here and
    # 0.0.0.0 inside a container, so the same call works in both places.
    app.run(port=int(os.environ.get("PORT", "8080")))
