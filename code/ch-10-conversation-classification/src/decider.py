# /// script
# dependencies = ["boto3", "botocore[crt]"]
# ///
"""Provide three interchangeable clients for a decision model.

Every client takes a state and typed questions and returns structured answers,
so the router in decide.py never knows which one it has.

- local: the model loaded into this process, the default for a laptop round.
- http: any System One server. Strands Decider and Kev both serve
  POST /v1/systemone with the same request and response, so swapping one for
  the other is a change of URL and nothing else.
- agentcore: an AgentCore Runtime endpoint, which carries the same payload.

Select with DECIDER_BACKEND. The response names the model that answered, so a
predictions file records what produced it and two runs stay comparable.

Usage:
  DECIDER_BACKEND=local  PYTHONPATH=src uv run src/decide.py --limit 40
  DECIDER_BACKEND=http   DECIDER_URL=http://127.0.0.1:8009 ... src/decide.py
"""

import json
import os
import urllib.parse
import urllib.request
import uuid

import boto3

BACKEND = os.environ.get("DECIDER_BACKEND", "local")
# Checkpoint identifier passed to the inference engine by the local backend.
DECIDER_MODEL = os.environ.get(
    "DECIDER_MODEL", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
)
# Leave empty for device selection, or set an explicit torch device.
DECIDER_DEVICE = os.environ.get("DECIDER_DEVICE", "")
DECIDER_URL = os.environ.get("DECIDER_URL", "")
DECIDER_TIMEOUT = int(os.environ.get("DECIDER_TIMEOUT", "300"))
AGENT_RUNTIME_ARN = os.environ.get("AGENT_RUNTIME_ARN", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# /v1/systemone is what both Strands Decider and Kev serve, so it is the default.
# Strands Decider also serves /invocations and /ping, which is the contract
# AgentCore Runtime invokes a container on, so pointing DECIDER_PATH at
# /invocations runs the local round through the same path the cloud round uses.
SYSTEM_ONE_PATH = os.environ.get("DECIDER_PATH") or "/v1/systemone"


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
    """The decision model loaded into this process."""

    def __init__(self) -> None:
        """Load the model checkpoint on the selected device."""
        from strands_decider.infer import load_engine  # lazy: local path only

        device = best_device()
        self.where = f"local:{device}"
        self._engine = load_engine(DECIDER_MODEL, device=device)

    def decide(self, state: str, questions: dict) -> dict:
        """Answer every question about one state in a single forward pass."""
        # pydantic coerces the plain question dicts into the typed primitives,
        # so the same dicts travel to the other two backends unchanged
        return self._engine.ask(state, questions).model_dump()


class HttpDecider:
    """A decision model behind a System One server, Strands Decider or Kev.

    Both projects serve the same endpoint and the same request and response, so
    this one client reaches either. Point DECIDER_URL at whichever is running:

      uv run --with strands-decider strands-decider serve <checkpoint> --port 8009
      uv run -m kev.serve --run jaredpalmer/kev-4b --port 8009
    """

    def __init__(self, url: str = "") -> None:
        """Hold the server's base URL, refusing to guess one."""
        url = url or DECIDER_URL
        if not url:
            raise SystemExit(
                "set DECIDER_URL to a System One server, for example "
                "http://127.0.0.1:8009, or DECIDER_BACKEND=local to run the "
                "model in this process"
            )
        scheme = urllib.parse.urlparse(url).scheme
        if scheme not in ("http", "https"):
            # urlopen will happily take file:// and other schemes; a decision
            # model is reached over HTTP and nothing else
            raise SystemExit(f"DECIDER_URL must be http or https, not {scheme!r}")
        self._url = url.rstrip("/") + SYSTEM_ONE_PATH
        self.where = url.rstrip("/")

    def decide(self, state: str, questions: dict) -> dict:
        """Post one System One request and return the answers."""
        body = json.dumps({"state": state, "questions": questions}).encode()
        request = urllib.request.Request(
            self._url, data=body, headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=DECIDER_TIMEOUT) as response:
            return json.loads(response.read())


class AgentCoreDecider:
    """The decision model behind an AgentCore Runtime endpoint."""

    def __init__(self) -> None:
        """Hold the runtime ARN and one session id for the whole run."""
        if not AGENT_RUNTIME_ARN:
            raise SystemExit(
                "set AGENT_RUNTIME_ARN to the runtime from `make deploy`, "
                "or DECIDER_BACKEND=local to run the model in this process"
            )
        self.where = "agentcore"
        # One session for the whole run: the first call provisions the instance
        # and loads the weights, and later calls carrying the same session id
        # land on that warm instance. The service wants 33 to 100 characters.
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


BACKENDS = {"local": LocalDecider, "http": HttpDecider, "agentcore": AgentCoreDecider}


def get_decider():
    """Return the decision model named by DECIDER_BACKEND."""
    try:
        return BACKENDS[BACKEND]()
    except KeyError:
        raise SystemExit(
            f"DECIDER_BACKEND={BACKEND!r} is not one of {', '.join(BACKENDS)}"
        ) from None
