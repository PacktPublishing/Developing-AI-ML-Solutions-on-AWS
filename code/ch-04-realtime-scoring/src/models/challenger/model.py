"""The challenger model: a second logistic PD with different weights.

The hypothesis the ensemble gateway averages against the champion scorecard. Same
feature contract, different coefficients: it leans harder on revolving
utilization and recent delinquency, lighter on debt-to-income.
"""

import math

INTERCEPT = -3.2
COEF_DTI = 1.8
COEF_UTILIZATION = 4.0
COEF_AMOUNT_TO_INCOME = 1.2
COEF_DAYS_PAST_DUE = 0.12


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
