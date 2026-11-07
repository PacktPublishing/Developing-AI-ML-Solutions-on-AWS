# /// script
# dependencies = ["boto3", "botocore[crt]"]
# ///
"""Invoke the chapter's AgentCore runtime with one JSON payload.

Used for the calls that are not routing: preparing the volume, and checking by
hand what a deployed runtime answers. The router itself goes through
src/decider.py, which carries one session id across a whole run.

Usage:
  uv run aws/invoke.py --arn <runtime-arn> --payload '{"mode": "prepare"}'
"""

import argparse
import json
import sys
import uuid

import boto3
from botocore.config import Config


def main() -> None:
    """Invoke the runtime once and print what came back."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arn", required=True, help="the runtime ARN from the stack")
    p.add_argument("--payload", required=True, help="the request, as JSON")
    p.add_argument("--region", default="us-east-1")
    p.add_argument(
        "--session",
        default="",
        help="reuse a session to land on its warm instance; a new one provisions",
    )
    a = p.parse_args()

    try:
        payload = json.loads(a.payload)
    except json.JSONDecodeError as err:
        raise SystemExit(f"--payload is not JSON: {err}") from None

    # The service wants 33 to 100 characters.
    session = a.session or f"ch10-invoke-{uuid.uuid4()}-{uuid.uuid4().hex[:8]}"
    client = boto3.client(
        "bedrock-agentcore",
        region_name=a.region,
        # Preparing a volume installs torch and pulls 4.8 GB of weights, and the
        # first call of a session also waits for an instance, so the read
        # timeout is the service's own 15 minute request ceiling.
        config=Config(
            read_timeout=900, retries={"max_attempts": 2, "mode": "standard"}
        ),
    )
    print(f"session {session}", file=sys.stderr)
    response = client.invoke_agent_runtime(
        agentRuntimeArn=a.arn,
        qualifier="DEFAULT",
        runtimeSessionId=session,
        payload=json.dumps(payload).encode(),
    )
    body = response["response"].read()
    try:
        print(json.dumps(json.loads(body), indent=2))
    except json.JSONDecodeError:
        # an agent that answered with something other than JSON is still worth seeing
        print(body.decode("utf-8", "replace"))


if __name__ == "__main__":
    main()
