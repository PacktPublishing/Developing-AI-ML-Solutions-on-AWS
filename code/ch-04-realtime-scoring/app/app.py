"""The decision service: a loan application in, a decision out.

The model gives a probability of default; the policy turns that, together
with hard rules, into approve, refer, or decline, and attaches the reason
codes an adverse-action notice needs. The hard rules run first because a
failed KYC check must never be overruled by a good score.

The service knows nothing about AWS: it runs anywhere a container does.
When DECISIONS_TABLE is set, every decision is also logged to DynamoDB —
DynamoDB Local on the laptop, the real table in the cloud, the same boto3
call either way.
"""

import os

import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from scorecard import probability_of_default

DECISIONS_TABLE = os.environ.get("DECISIONS_TABLE", "")
DYNAMODB_ENDPOINT = os.environ.get("DYNAMODB_ENDPOINT", "")

app = FastAPI(title="decision-service")
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
    """The decision and the reasons it can be defended with."""

    application_id: str
    decision: str  # APPROVE | REFER | DECLINE
    pd: float
    score: int
    reasons: list[str]


def decide(item: Application) -> DecisionResponse:
    """Apply hard policy rules first, then the score cutoff."""
    reasons = []
    if not item.kyc_passed:
        reasons.append("KYC_FAILED")
    if item.age < 18:
        reasons.append("UNDER_MINIMUM_AGE")
    if item.dti > 0.50:
        reasons.append("DTI_TOO_HIGH")
    if item.requested_amount > 10 * item.monthly_income:
        reasons.append("AMOUNT_EXCEEDS_POLICY")

    pd = probability_of_default(item)
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
        score=score,
        reasons=reasons,
    )


def _log(response: DecisionResponse) -> None:
    """Write the decision to the log table, if one is configured."""
    global _table
    if not DECISIONS_TABLE:
        return
    if _table is None:
        _table = boto3.resource(
            "dynamodb", endpoint_url=DYNAMODB_ENDPOINT or None
        ).Table(DECISIONS_TABLE)
    item = response.model_dump()
    item["pd"] = str(item["pd"])  # DynamoDB numbers travel as Decimal; keep it simple
    _table.put_item(Item=item)


@app.post("/decision", response_model=DecisionResponse)
def decision(application: Application) -> DecisionResponse:
    """Decide one application."""
    response = decide(application)
    _log(response)
    return response


@app.get("/health")
def health() -> dict:
    """Answer the ALB target-group health check."""
    return {"status": "ok"}
