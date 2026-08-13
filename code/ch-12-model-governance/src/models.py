"""The two models the router routes between, behind one scoring seam.

score(model_name, loan) returns the probability of default from whichever model the
rollout picked. Both champion (credit-scorecard) and challenger (credit-challenger)
speak the same contract from chapter 2: POST a JSON application, read back {"pd": [p]}.
Locally they are two serving containers reached by URL; on AWS they are two SageMaker
endpoints reached by InvokeEndpoint. get_scorer() picks the path by environment.

Env: CHAMPION_URL / CHALLENGER_URL (local), or CHAMPION_ENDPOINT / CHALLENGER_ENDPOINT
     (SageMaker endpoint names) plus AWS_REGION.
"""

import json
import os
import urllib.request

URLS = {"credit-scorecard": "CHAMPION_URL", "credit-challenger": "CHALLENGER_URL"}
ENDPOINTS = {
    "credit-scorecard": "CHAMPION_ENDPOINT",
    "credit-challenger": "CHALLENGER_ENDPOINT",
}


def _pd(body: dict) -> float:
    """Read the probability of default from a serving response."""
    pd = body["pd"]
    return float(pd[0] if isinstance(pd, list) else pd)


def http_scorer():
    """Return score(model_name, loan) that POSTs to the model's /invocations URL (local)."""
    urls = {name: os.environ[env] for name, env in URLS.items()}

    def score(model_name: str, loan: dict) -> float:
        req = urllib.request.Request(
            urls[model_name],
            data=json.dumps(loan).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return _pd(json.loads(resp.read()))

    return score


def sagemaker_scorer():
    """Return score(model_name, loan) that invokes the model's SageMaker endpoint (AWS)."""
    import boto3

    runtime = boto3.client(
        "sagemaker-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1")
    )
    endpoints = {name: os.environ[env] for name, env in ENDPOINTS.items()}

    def score(model_name: str, loan: dict) -> float:
        resp = runtime.invoke_endpoint(
            EndpointName=endpoints[model_name],
            ContentType="application/json",
            Body=json.dumps(loan),
        )
        return _pd(json.loads(resp["Body"].read()))

    return score


def get_scorer():
    """Return the scorer: SageMaker endpoints when configured, else the local URLs."""
    if os.environ.get("CHAMPION_ENDPOINT"):
        return sagemaker_scorer()
    return http_scorer()
