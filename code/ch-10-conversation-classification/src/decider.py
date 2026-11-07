# /// script
# dependencies = ["boto3"]
# ///
"""The decision-model seam: the model in this process, or on AgentCore Runtime.

A decision model is handed a state and a typed question and answers by scoring
the options it was given. There is no text to generate, so there is no answer to
parse: a choice question comes back as the chosen option, a probability for every
option, and a confidence. Both backends speak the same request and response,
because the Strands Decider server serves `POST /invocations` and `GET /ping`,
which is also the contract AgentCore Runtime invokes a container on.

Usage:
  DECIDER_LOCAL=1 PYTHONPATH=src uv run src/decide.py --limit 40
"""

import json
import os
import uuid

import boto3

# The released checkpoint carries a LoRA adapter and the pointer head, about
# 0.2 GB; the Qwen3.5-2B-Base torso it sits on is another 4.5 GB, pulled on
# first load. That split is why the AWS side mounts a volume for weights rather
# than baking them into an image.
DECIDER_MODEL = os.environ.get(
    "DECIDER_MODEL", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
)
# Empty means pick the best device present. Set it to cpu to feel what the
# chapter's figures look like without a GPU.
DECIDER_DEVICE = os.environ.get("DECIDER_DEVICE", "")
AGENT_RUNTIME_ARN = os.environ.get("AGENT_RUNTIME_ARN", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def best_device() -> str:
    """Return the fastest torch device on this machine.

    Apple silicon reports mps, an NVIDIA box reports cuda, and anything else
    gets cpu, where a 2B torso runs but slowly enough to notice.
    """
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
        """Load the checkpoint once; the weights are the expensive part."""
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
        # A long first call provisions the instance and loads the weights; every
        # later call carrying the same session id lands on that warm instance,
        # so the session is created once per run and not once per conversation.
        # The service requires 33 to 100 characters.
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
