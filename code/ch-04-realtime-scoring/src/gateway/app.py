"""The decision gateway: a FastAPI service that turns several model endpoints into one loan decision.

It fans each application out to every endpoint in ENDPOINTS, averages the PDs of the deciding
models, applies the KYC/policy rules, and logs the decision to DynamoDB. A http target is a model
container (local); anything else is a SageMaker endpoint name (cloud).

Two knobs separate the two patterns this can serve. SHADOW_MODELS names endpoints that are called
and logged but kept out of the average, which is champion/challenger: the challenger is measured on
live traffic without moving the decision. Leave it empty and every model contributes equally, which
is an ensemble. MODEL_TIMEOUT and MODEL_RETRIES bound each call; a deciding model that cannot be
reached sends the application to a human rather than deciding it on partial evidence.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import httpx
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI
from pydantic import BaseModel

# name -> target (a http URL locally, a SageMaker endpoint name on AWS)
ENDPOINTS = dict(
    pair.split("=", 1)
    for pair in os.environ.get("ENDPOINTS", "").split(",")
    if "=" in pair
)
DECISIONS_TABLE = os.environ.get("DECISIONS_TABLE", "")
DYNAMODB_ENDPOINT = os.environ.get("DYNAMODB_ENDPOINT", "")

# Models called and logged but excluded from the decision (champion/challenger).
# Empty by default, which makes every endpoint a deciding model (an ensemble).
SHADOW_MODELS = {
    n.strip() for n in os.environ.get("SHADOW_MODELS", "").split(",") if n.strip()
}
# A decision is answered in front of a waiting applicant, so a slow model is a failed model.
MODEL_TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "2.0"))
MODEL_ATTEMPTS = int(os.environ.get("MODEL_ATTEMPTS", "2"))

app = FastAPI(title="decision-gateway")
_runtime = None
_table = None
# One pooled client for the whole process. httpx.post() would open a fresh TCP
# connection per call, which costs more than the model inference itself.
_http = httpx.Client(timeout=MODEL_TIMEOUT)


class Application(BaseModel):
    """One loan application, the request contract."""

    application_id: str
    age: int
    monthly_income: float
    requested_amount: float
    dti: float  # debt-to-income, 0..1
    utilization: float  # revolving utilization, 0..1
    days_past_due: int = 0
    kyc_passed: bool = True


class DecisionResponse(BaseModel):
    """The decision, the deciding PD, and every model's PD behind it."""

    application_id: str
    decision: str  # APPROVE | REFER | DECLINE
    pd: float  # mean of the deciding models; 0.0 when none could be reached
    model_pds: dict[str, float]  # every model that answered, shadows included
    unavailable: list[str]  # models that did not answer
    score: int
    reasons: list[str]


def _once(target: str, body: str) -> float:
    """One scoring call (http target = local container, else a SageMaker endpoint)."""
    if target.startswith("http"):
        resp = _http.post(
            f"{target}/invocations",
            content=body,
            headers={"content-type": "application/json"},
        )
        resp.raise_for_status()
        return float(resp.json()["pd"])
    global _runtime
    if _runtime is None:
        # botocore retries and waits forever by default; both are bounded here so a
        # stalled endpoint cannot hold the request open past the decision budget.
        _runtime = boto3.client(
            "sagemaker-runtime",
            config=Config(
                connect_timeout=MODEL_TIMEOUT,
                read_timeout=MODEL_TIMEOUT,
                retries={"max_attempts": 1, "mode": "standard"},
            ),
        )
    resp = _runtime.invoke_endpoint(
        EndpointName=target, ContentType="application/json", Body=body
    )
    return float(json.loads(resp["Body"].read())["pd"])


def _invoke(target: str, features: dict) -> float:
    """Score at one endpoint, retrying a bounded number of times before giving up."""
    body = json.dumps(features)
    # everything a model call can fail with: the transport (timeout, refused, 5xx)
    # and the payload (missing "pd", not a number)
    failures = (
        httpx.HTTPError,
        BotoCoreError,
        ClientError,
        KeyError,
        TypeError,
        ValueError,
    )
    last: Exception | None = None
    for _ in range(MODEL_ATTEMPTS):
        try:
            return _once(target, body)
        except failures as exc:
            last = exc
    raise RuntimeError(f"{target} did not answer") from last


def _score(item: Application) -> tuple[float | None, dict[str, float], list[str]]:
    """Call every model; average the deciding ones, keep the shadows out of the number."""
    features = item.model_dump()
    model_pds: dict[str, float] = {}
    unavailable: list[str] = []
    # Call the models at the same time, not one after another. Both calls block on the
    # network, so the request waits for the slowest model rather than for their sum,
    # which is the difference between one latency budget and two.
    with ThreadPoolExecutor(max_workers=max(1, len(ENDPOINTS))) as pool:
        pending = {
            pool.submit(_invoke, target, features): name
            for name, target in ENDPOINTS.items()
        }
        for future, name in pending.items():
            try:
                model_pds[name] = future.result()
            except RuntimeError:
                unavailable.append(name)
    deciding = {n: v for n, v in model_pds.items() if n not in SHADOW_MODELS}
    pd = sum(deciding.values()) / len(deciding) if deciding else None
    return pd, model_pds, unavailable


def decide(item: Application) -> DecisionResponse:
    """Hard policy rules first, then the score cutoff, with a human catching the gaps."""
    reasons = []
    if not item.kyc_passed:
        reasons.append("KYC_FAILED")
    if item.age < 18:
        reasons.append("UNDER_MINIMUM_AGE")
    if item.dti > 0.50:
        reasons.append("DTI_TOO_HIGH")
    if item.requested_amount > 10 * item.monthly_income:
        reasons.append("AMOUNT_EXCEEDS_POLICY")

    pd, model_pds, unavailable = _score(item)
    deciding_down = [n for n in unavailable if n not in SHADOW_MODELS]
    score = round((1.0 - pd) * 1000) if pd is not None else 0

    if reasons:
        # A hard rule needs no model: the application fails on its own terms.
        decision = "DECLINE"
    elif deciding_down or pd is None:
        # Never approve or decline on partial evidence. A model the gateway could not
        # reach is a gap in the file, and a gap in the file is a person's job. pd is
        # None only if nothing was left to decide with, which is a misconfiguration
        # (every endpoint shadowed) rather than an outage, so it is named separately.
        decision = "REFER"
        reasons.append("MODEL_UNAVAILABLE" if deciding_down else "NO_DECIDING_MODEL")
    elif pd < 0.10:
        decision = "APPROVE"
    elif pd < 0.20:
        decision = "REFER"
        reasons.append("BORDERLINE_MANUAL_REVIEW")
    else:
        decision = "DECLINE"
        reasons.append("SCORE_BELOW_CUTOFF")
    return DecisionResponse(
        application_id=item.application_id,
        decision=decision,
        pd=round(pd, 4) if pd is not None else 0.0,
        model_pds={name: round(v, 4) for name, v in model_pds.items()},
        unavailable=unavailable,
        score=score,
        reasons=reasons,
    )


def _log(response: DecisionResponse) -> None:
    """Write the decision (and every model's PD) to the log table, if configured."""
    global _table
    if not DECISIONS_TABLE:
        return
    if _table is None:
        _table = boto3.resource(
            "dynamodb", endpoint_url=DYNAMODB_ENDPOINT or None
        ).Table(DECISIONS_TABLE)
    item = response.model_dump()
    item["pd"] = str(item["pd"])
    item["model_pds"] = {k: str(v) for k, v in item["model_pds"].items()}
    _table.put_item(Item=item)


@app.post("/decision", response_model=DecisionResponse)
def decision(application: Application) -> DecisionResponse:
    """Decide one application across the model ensemble."""
    response = decide(application)
    _log(response)
    return response


@app.get("/health")
def health() -> dict:
    """Answer the ALB / target-group health check."""
    return {"status": "ok"}
