"""The scorecard behind the decision service.

A transparent logistic scorecard standing in for a trained artifact you
would load from S3 once at startup: the coefficients are written out so the
chapter can point at every term when it explains reason codes. Swap this
for a real model — the ch-02 scorecard container, a SageMaker endpoint —
and nothing else in the service changes: the policy, the reasons, and the
contract stay put.
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app import Application

# One coefficient per risk driver; positive pushes toward default. The
# amount term is the requested amount against a year of income — asking for
# more than you earn is a different loan than topping up a paycheck.
INTERCEPT = -3.5
COEF_DTI = 2.5
COEF_UTILIZATION = 3.0
COEF_AMOUNT_TO_INCOME = 1.5
COEF_DAYS_PAST_DUE = 0.05


def probability_of_default(item: "Application") -> float:
    """Score one application: a logit over the application's risk drivers."""
    amount_to_annual_income = item.requested_amount / max(
        item.monthly_income * 12.0, 1.0
    )
    z = (
        INTERCEPT
        + COEF_DTI * item.dti
        + COEF_UTILIZATION * item.utilization
        + COEF_AMOUNT_TO_INCOME * amount_to_annual_income
        + COEF_DAYS_PAST_DUE * item.days_past_due
    )
    return 1.0 / (1.0 + math.exp(-z))
