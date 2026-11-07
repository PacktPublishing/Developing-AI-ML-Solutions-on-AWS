"""The routing agent: owns the policy, asks the model.

AgentCore Runtime hosts agents and SageMaker hosts models, and this chapter
splits them that way. The decision model sits behind a SageMaker endpoint on a
GPU instance, because it is 2 GB of image and 4.5 GB of weights. This agent is a
few kilobytes on a serverless microVM, and it holds everything about routing
that is not the model: which teams exist, where their work begins and ends, and
how sure the model has to be before a conversation is routed rather than left
for a person.

That split is why AgentCore works here at all. An agent that calls an endpoint
fits a microVM easily; the model never would, at a 2 GB image limit and an 8 GB
session limit against 7.4 GB of CPU inference.

A caller sends the conversation and gets a routing decision:

  {"text": "<the conversation>", "variant": "b", "threshold": 0.6}
  -> {"team": "Cards", "confidence": 0.95, "routed": true, "probabilities": {...}}

A full System One request is passed through untouched, so the same runtime can
be used to ask the model something the policy does not cover.
"""

import json
import os

import boto3
import categories
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from botocore.exceptions import ClientError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "")
ASK = "Which team should handle this conversation?"

app = BedrockAgentCoreApp()
_runtime = None


def runtime():
    """Return the SageMaker runtime client, made once."""
    global _runtime
    if _runtime is None:
        _runtime = boto3.client("sagemaker-runtime")
    return _runtime


def ask_model(request: dict) -> dict:
    """Send one System One request to the endpoint and return its answer."""
    if not ENDPOINT_NAME:
        return {"error": "ENDPOINT_NAME is not set, so there is no model to ask"}
    try:
        response = runtime().invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(request).encode(),
        )
    except ClientError as err:
        # An endpoint that is missing, still creating, or refusing the call is
        # the common case while a round is being set up. Answering with the
        # reason beats a 500 that sends the reader to CloudWatch.
        return {"error": f"the endpoint {ENDPOINT_NAME} did not answer: {err}"}
    return json.loads(response["Body"].read())


@app.entrypoint
def route(payload: dict) -> dict:
    """Route one conversation, or pass a System One request through.

    The SDK hands the payload over unchanged, so it is checked here. A bad
    request is answered with the reason rather than a stack trace, because the
    caller is a loop over someone's conversations and one bad request should not
    read like the model failing.
    """
    if not isinstance(payload, dict):
        return {"error": "the payload must be a JSON object"}

    # A caller that built its own question gets it asked as written.
    if isinstance(payload.get("questions"), dict) and payload["questions"]:
        return ask_model(payload)

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "send text, the conversation to route, or a questions object"}

    labels = payload.get("labels") or list(categories.BORDERS)
    if not isinstance(labels, list) or len(labels) < 2:
        return {"error": "labels must be a list of at least two teams"}
    described = payload.get("variant", "b") == "b"
    threshold = payload.get("threshold", 0.0)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        return {"error": "threshold must be a number"}

    answered = ask_model(
        {
            "state": text,
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": ASK,
                    "criteria": categories.as_criteria(labels, described),
                }
            },
        }
    )
    if "error" in answered:
        return answered
    answer = answered["answers"]["team"]
    confidence = answer["confidence"]
    return {
        "team": answer["choice"],
        "probabilities": answer["probabilities"],
        "confidence": confidence,
        # Below the threshold nothing is routed. That is not the model failing:
        # it reported a close call, and the policy is that close calls go to a
        # person. The policy lives here rather than in the model.
        "routed": confidence >= threshold,
        "model": answered.get("model", "unnamed"),
    }


if __name__ == "__main__":
    # 8080 is the port AgentCore invokes, and the SDK binds 0.0.0.0 in a
    # container and 127.0.0.1 here.
    app.run(port=int(os.environ.get("PORT", "8080")))
