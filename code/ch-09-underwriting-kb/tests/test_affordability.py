"""Affordability arithmetic: the instalment, the ratio, and the inverse."""

import pytest
from affordability import dti, instalment, max_affordable_amount


def test_zero_rate_is_straight_division():
    """With no interest the instalment is the amount spread over the tenor."""
    assert instalment(120_000, 0.0, 12) == pytest.approx(10_000.0)


def test_instalment_grows_with_rate():
    """A higher rate cannot produce a smaller payment over the same tenor."""
    assert instalment(300_000, 24.0, 12) > instalment(300_000, 18.0, 12)


def test_dti_counts_existing_repayments():
    """The ratio is existing plus proposed over income, not the new loan alone."""
    without = dti(100_000, 0, 200_000, 18.0, 12).dti_percent
    with_existing = dti(100_000, 10_000, 200_000, 18.0, 12).dti_percent
    assert with_existing == pytest.approx(without + 10.0, abs=0.1)


def test_recommended_amount_lands_on_the_ceiling():
    """The inverse is exact: lending the maximum puts DTI at the ceiling."""
    amount = max_affordable_amount(90_000, 12_000, 18.0, 12, ceiling_percent=40.0)
    assert dti(90_000, 12_000, amount, 18.0, 12).dti_percent == pytest.approx(40.0)


def test_no_headroom_returns_zero():
    """A borrower already past the ceiling can afford nothing further."""
    assert max_affordable_amount(50_000, 25_000, 18.0, 12, ceiling_percent=40.0) == 0.0


def test_income_and_tenor_are_validated():
    """Impossible inputs raise rather than returning a number nobody should use."""
    with pytest.raises(ValueError):
        dti(0, 0, 100_000, 18.0, 12)
    with pytest.raises(ValueError):
        instalment(100_000, 18.0, 0)
