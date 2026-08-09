"""The two models the router routes between, behind one scoring seam.

score(model_name, loan) returns the probability of default from whichever model the
rollout picked. Both champion (credit-scorecard) and challenger (credit-challenger)
speak the same /invocations contract from chapter 2 -- POST a JSON application, read
back {"pd": [p]} -- so the router treats them identically and only the URL changes.

Env: CHAMPION_URL, CHALLENGER_URL (the two /invocations endpoints).
"""

import json
import os
import urllib.request

MODELS = {
    "credit-scorecard": "CHAMPION_URL",
    "credit-challenger": "CHALLENGER_URL",
}


def _invoke(url: str, loan: dict) -> float:
    """POST one application to a serving endpoint and read back its probability."""
    req = urllib.request.Request(
        url,
        data=json.dumps(loan).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    pd = body["pd"]
    return float(pd[0] if isinstance(pd, list) else pd)


def http_scorer():
    """Return score(model_name, loan) that routes to the model's endpoint URL."""
    urls = {name: os.environ[env] for name, env in MODELS.items()}

    def score(model_name: str, loan: dict) -> float:
        return _invoke(urls[model_name], loan)

    return score
