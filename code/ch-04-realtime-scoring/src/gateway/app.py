"""The decision gateway: one FastAPI service that orchestrates several model endpoints.

It fans a loan application out to every configured endpoint, averages
their probabilities of default into one ensemble PD, turns that (with the hard
KYC/policy rules) into approve / refer / decline, and logs the decision — plus
each model's PD — to DynamoDB.

The gateway knows nothing about where the models run. Each endpoint in ENDPOINTS
is `name=target`; a target that starts with http is a model container's URL
(SageMaker local mode on the laptop), anything else is a SageMaker endpoint name
invoked with sagemaker-runtime (the cloud). Same ensemble either way.
"""

import json
import os

import boto3
import httpx
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

app = FastAPI(title="decision-gateway")
_runtime = None
_table = None


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
    """The decision, the ensemble PD, and each model's PD behind it."""

    application_id: str
    decision: str  # APPROVE | REFER | DECLINE
    pd: float  # ensemble
    model_pds: dict[str, float]
    score: int
    reasons: list[str]


def _invoke(target: str, features: dict) -> float:
    """Score the features at one endpoint and return its PD.

    A http target is a model container's /invocations (SageMaker local mode); any
    other target is a SageMaker endpoint name invoked with sagemaker-runtime.
    """
    body = json.dumps(features)
    if target.startswith("http"):
        resp = httpx.post(
            f"{target}/invocations",
            content=body,
            headers={"content-type": "application/json"},
            timeout=15,
        )
        return float(resp.json()["pd"])
    global _runtime
    if _runtime is None:
        _runtime = boto3.client("sagemaker-runtime")
    resp = _runtime.invoke_endpoint(
        EndpointName=target, ContentType="application/json", Body=body
    )
    return float(json.loads(resp["Body"].read())["pd"])


def _ensemble(item: Application) -> tuple[float, dict[str, float]]:
    """Invoke every model and average their PDs."""
    features = item.model_dump()
    model_pds = {name: _invoke(target, features) for name, target in ENDPOINTS.items()}
    return sum(model_pds.values()) / len(model_pds), model_pds


def decide(item: Application) -> DecisionResponse:
    """Hard policy rules first, then the ensemble score cutoff."""
    reasons = []
    if not item.kyc_passed:
        reasons.append("KYC_FAILED")
    if item.age < 18:
        reasons.append("UNDER_MINIMUM_AGE")
    if item.dti > 0.50:
        reasons.append("DTI_TOO_HIGH")
    if item.requested_amount > 10 * item.monthly_income:
        reasons.append("AMOUNT_EXCEEDS_POLICY")

    pd, model_pds = _ensemble(item)
    score = round((1.0 - pd) * 1000)

    if reasons:
        decision = "DECLINE"
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
        pd=round(pd, 4),
        model_pds={name: round(v, 4) for name, v in model_pds.items()},
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
