"""The scorecard model: a transparent logistic PD over the risk drivers.

One coefficient per driver, written out so the chapter can point at every term.
This is the "champion" the ensemble gateway invokes as one endpoint.
"""

import math

INTERCEPT = -3.5
COEF_DTI = 2.5
COEF_UTILIZATION = 3.0
COEF_AMOUNT_TO_INCOME = 1.5
COEF_DAYS_PAST_DUE = 0.05


def probability_of_default(features: dict) -> float:
    """Score one application from its feature dict: a logit over the risk drivers."""
    amount_to_annual_income = features["requested_amount"] / max(
        features["monthly_income"] * 12.0, 1.0
    )
    z = (
        INTERCEPT
        + COEF_DTI * features["dti"]
        + COEF_UTILIZATION * features["utilization"]
        + COEF_AMOUNT_TO_INCOME * amount_to_annual_income
        + COEF_DAYS_PAST_DUE * features.get("days_past_due", 0)
    )
    return 1.0 / (1.0 + math.exp(-z))
