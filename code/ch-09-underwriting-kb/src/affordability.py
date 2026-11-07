"""Debt-to-income arithmetic for the underwriting agent.

The knowledge base answers what was decided before. This answers what the
numbers allow now: the instalment a loan implies, the DTI that produces, and
the largest amount that still sits inside a DTI ceiling.

Kept free of any model or store so it can be unit-tested on its own, and so the
agent tool wrapping it has nothing to do but call it. A model fills these
arguments in from a conversation, so every entry point validates the same way
through the checks below: a negative principal has no instalment, and a
negative tenor has no largest affordable amount, and both are easier to catch
here than to recognise in an answer.

Usage:
    from affordability import dti, max_affordable_amount
"""

import math
import os
from dataclasses import dataclass

# The debt-to-income ceiling is credit policy, not a per-question choice. It lives
# here so a caller cannot supply its own, which an agent will otherwise do.
CEILING_PERCENT = float(os.environ.get("DTI_CEILING_PERCENT", "40"))

# Outer bounds, wide enough that no real request meets them and narrow enough
# that a misread figure or a model's placeholder does.
MAX_AMOUNT = 1e12
MAX_RATE_PERCENT = 200.0
MAX_TENOR_MONTHS = 600


@dataclass(frozen=True)
class Affordability:
    """One affordability assessment, in the units the memos use."""

    instalment: float
    dti_percent: float
    within_ceiling: bool
    headroom: float


# -------------------------------------------------------------------------------
# Input validation, shared by every entry point
# -------------------------------------------------------------------------------
def _number(name: str, value: float) -> float:
    """Return the value as a finite float, or raise naming the argument."""
    try:
        value = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a number") from err
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _months(tenor_months: int) -> int:
    """Return the tenor as a whole number of months inside the allowed range.

    A tenor is a count of payments, so 12.5 months is not a shorter loan, it is
    a mistake. An integral float passes because a model's JSON says 12.0.
    """
    # True is an integer in Python and would pass as a one month tenor
    if tenor_months is True or tenor_months is False:
        raise ValueError("tenor_months must be a whole number of months")
    value = _number("tenor_months", tenor_months)
    if not value.is_integer():
        raise ValueError("tenor_months must be a whole number of months")
    months = int(value)
    if not 1 <= months <= MAX_TENOR_MONTHS:
        raise ValueError(f"tenor_months must be 1 to {MAX_TENOR_MONTHS}")
    return months


def _terms(amount: float, annual_rate_percent: float, tenor_months: int) -> tuple:
    """Validate one proposed facility: its amount, its rate, and its tenor."""
    amount = _number("amount", amount)
    if not 0 <= amount <= MAX_AMOUNT:
        raise ValueError(f"amount must be 0 to {MAX_AMOUNT:,.0f}")
    rate = _number("annual_rate_percent", annual_rate_percent)
    if not 0 <= rate <= MAX_RATE_PERCENT:
        raise ValueError(f"annual_rate_percent must be 0 to {MAX_RATE_PERCENT:,.0f}")
    return amount, rate, _months(tenor_months)


def _capacity(
    monthly_income: float, existing_repayments: float, ceiling_percent: float
) -> tuple:
    """Validate the borrower's side: income, what they already pay, the ceiling."""
    income = _number("monthly_income", monthly_income)
    if income <= 0:
        raise ValueError("monthly_income must be positive")
    repayments = _number("existing_repayments", existing_repayments)
    if repayments < 0:
        raise ValueError("existing_repayments cannot be negative")
    ceiling = _number("ceiling_percent", ceiling_percent)
    if not 0 < ceiling <= 100:
        raise ValueError("ceiling_percent must be above 0 and at most 100")
    return income, repayments, ceiling


def instalment(amount: float, annual_rate_percent: float, tenor_months: int) -> float:
    """Return the level monthly payment amortising `amount` over `tenor_months`."""
    amount, rate, months = _terms(amount, annual_rate_percent, tenor_months)
    monthly_rate = rate / 100.0 / 12.0
    if monthly_rate == 0:
        return amount / months
    factor = (1 + monthly_rate) ** months
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
    income, repayments, ceiling = _capacity(
        monthly_income, existing_repayments, ceiling_percent
    )
    payment = instalment(amount, annual_rate_percent, tenor_months)
    ratio = (repayments + payment) / income * 100.0
    allowed = income * ceiling / 100.0
    return Affordability(
        instalment=round(payment, 2),
        dti_percent=round(ratio, 1),
        within_ceiling=ratio <= ceiling,
        headroom=round(allowed - repayments - payment, 2),
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
    income, repayments, ceiling = _capacity(
        monthly_income, existing_repayments, ceiling_percent
    )
    # the amount is what this answers, so only the rate and the tenor are checked
    _, rate, months = _terms(0.0, annual_rate_percent, tenor_months)
    budget = income * ceiling / 100.0 - repayments
    if budget <= 0:
        return 0.0
    monthly_rate = rate / 100.0 / 12.0
    if monthly_rate == 0:
        return round(budget * months, 2)
    factor = (1 + monthly_rate) ** months
    return round(budget * (factor - 1) / (monthly_rate * factor), 2)
