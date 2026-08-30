"""Debt-to-income arithmetic for the underwriting agent.

The knowledge base answers what was decided before. This answers what the
numbers allow now: the instalment a loan implies, the DTI that produces, and
the largest amount that still sits inside a DTI ceiling.

Kept free of any model or store so it can be unit-tested on its own, and so the
agent tool wrapping it has nothing to do but call it.

Usage:
    from affordability import dti, max_affordable_amount
"""

import os
from dataclasses import dataclass

# The debt-to-income ceiling is credit policy, not a per-question choice. It lives
# here so a caller cannot supply its own, which an agent will otherwise do.
CEILING_PERCENT = float(os.environ.get("DTI_CEILING_PERCENT", "40"))


@dataclass(frozen=True)
class Affordability:
    """One affordability assessment, in the units the memos use."""

    instalment: float
    dti_percent: float
    within_ceiling: bool
    headroom: float


def instalment(amount: float, annual_rate_percent: float, tenor_months: int) -> float:
    """Return the level monthly payment amortising `amount` over `tenor_months`."""
    if tenor_months <= 0:
        raise ValueError("tenor_months must be positive")
    monthly_rate = annual_rate_percent / 100.0 / 12.0
    if monthly_rate == 0:
        return amount / tenor_months
    factor = (1 + monthly_rate) ** tenor_months
    return amount * monthly_rate * factor / (factor - 1)


# -------------------------------------------------------------------------------
# Debt-to-income ratio (DTI) and affordability
# -------------------------------------------------------------------------------
def dti(
    monthly_income: float,
    existing_repayments: float,
    amount: float,
    annual_rate_percent: float,
    tenor_months: int,
    ceiling_percent: float = 40.0,
) -> Affordability:
    """Assess one proposed loan against a DTI ceiling."""
    if monthly_income <= 0:
        raise ValueError("monthly_income must be positive")
    payment = instalment(amount, annual_rate_percent, tenor_months)
    ratio = (existing_repayments + payment) / monthly_income * 100.0
    allowed = monthly_income * ceiling_percent / 100.0
    return Affordability(
        instalment=round(payment, 2),
        dti_percent=round(ratio, 1),
        within_ceiling=ratio <= ceiling_percent,
        headroom=round(allowed - existing_repayments - payment, 2),
    )


# -------------------------------------------------------------------------------
# Maximum affordable amount for a given DTI ceiling
# -------------------------------------------------------------------------------
def max_affordable_amount(
    monthly_income: float,
    existing_repayments: float,
    annual_rate_percent: float,
    tenor_months: int,
    ceiling_percent: float = 40.0,
) -> float:
    """Return the largest amount whose instalment keeps DTI at or under the ceiling.

    Inverts the annuity formula rather than searching, so the answer is exact.
    """
    budget = monthly_income * ceiling_percent / 100.0 - existing_repayments
    if budget <= 0:
        return 0.0
    monthly_rate = annual_rate_percent / 100.0 / 12.0
    if monthly_rate == 0:
        return round(budget * tenor_months, 2)
    factor = (1 + monthly_rate) ** tenor_months
    return round(budget * (factor - 1) / (monthly_rate * factor), 2)
