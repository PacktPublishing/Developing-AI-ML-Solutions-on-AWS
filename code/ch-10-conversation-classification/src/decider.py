# /// script
# dependencies = ["boto3"]
# ///
"""Provide local and AgentCore clients for the decision model.

Both clients accept a state and typed questions and return structured answers.
The local client loads the inference engine; the AWS client invokes AgentCore.

Usage:
  DECIDER_LOCAL=1 PYTHONPATH=src uv run src/decide.py --limit 40
"""

import json
import os
import uuid

import boto3

# Checkpoint identifier passed to the inference engine.
DECIDER_MODEL = os.environ.get(
    "DECIDER_MODEL", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
)
# Leave empty for device selection, or set an explicit torch device.
DECIDER_DEVICE = os.environ.get("DECIDER_DEVICE", "")
AGENT_RUNTIME_ARN = os.environ.get("AGENT_RUNTIME_ARN", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def best_device() -> str:
    """Use DECIDER_DEVICE if set; otherwise select available CUDA, MPS or CPU."""
    if DECIDER_DEVICE:
        return DECIDER_DEVICE
    import torch  # lazy import: only the local backend needs torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalDecider:
    """The decision model loaded into this process, for the local round."""

    def __init__(self) -> None:
        """Load the model checkpoint on the selected device."""
        from strands_decider.infer import load_engine  # lazy: local path only

        self.device = best_device()
        self._engine = load_engine(DECIDER_MODEL, device=self.device)

    def decide(self, state: str, questions: dict) -> dict:
        """Answer every question about one state in a single forward pass."""
        # pydantic coerces the plain question dicts into the typed primitives,
        # so the same dicts travel to AgentCore unchanged
        return self._engine.ask(state, questions).model_dump()


class AgentCoreDecider:
    """The decision model behind an AgentCore Runtime endpoint."""

    def __init__(self) -> None:
        """Hold the runtime ARN and one session id for the whole run."""
        if not AGENT_RUNTIME_ARN:
            raise SystemExit(
                "set AGENT_RUNTIME_ARN to the runtime from `make deploy`, "
                "or DECIDER_LOCAL=1 to run the model here"
            )
        self.device = "agentcore"
        # Reuse a session ID across requests in this run.
        self._session = f"ch10-decider-{uuid.uuid4()}-{uuid.uuid4().hex[:8]}"
        self._client = boto3.client(
            "bedrock-agentcore",
            region_name=AWS_REGION,
            config=boto3.session.Config(
                read_timeout=900, retries={"max_attempts": 2, "mode": "standard"}
            ),
        )

    def decide(self, state: str, questions: dict) -> dict:
        """Invoke the runtime with one System One request and read the answer."""
        payload = json.dumps({"state": state, "questions": questions}).encode()
        response = self._client.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            qualifier="DEFAULT",
            runtimeSessionId=self._session,
            payload=payload,
        )
        return json.loads(response["response"].read())


def get_decider():
    """Return the decision model: in this process, or on AgentCore Runtime."""
    if os.environ.get("DECIDER_LOCAL") == "1":
        return LocalDecider()
    return AgentCoreDecider()
