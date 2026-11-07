"""Test income exclusions, signed amounts, extraction validation and masked errors."""

import pytest

pytest.importorskip("pdfplumber")

from statements import (
    NOT_CHECKED,
    RECONCILED,
    Summary,
    Transaction,
    Unreadable,
    _money,
    excluded_reason,
    reconcile,
    validate,
)


def summary(**kwargs) -> Summary:
    """Build a Summary with only the fields a test cares about."""
    fields = {
        "case_id": "CASE-20260000",
        "holder": "Elena Brennan",
        "months": 3,
        "total_inflows": 0.0,
        "monthly_income": 0.0,
        "existing_repayments": 0.0,
        "transactions": 12,
        "reconciliation": RECONCILED,
        "balance_breaks": [],
    }
    return Summary(**{**fields, **kwargs})


# ---------------------------------------------------------------------------
# Income is not the total of the credits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("description", "reason"),
    [
        ("TRANSFER FROM  OWN SAVINGS", "own-account transfer"),
        ("TRANSFER FROM  SELF CURRENT ACCOUNT", "own-account transfer"),
        ("TRANSFER IN  OWN ACCOUNT", "own-account transfer"),
        ("LOAN DISBURSEMENT  FACILITY 4471", "loan disbursement"),
        ("FACILITY ADVANCE", "loan disbursement"),
        ("DRAWDOWN  TERM LOAN", "loan disbursement"),
        ("ASSET SALE  DELIVERY VAN", "asset sale"),
        ("SALE OF  EQUIPMENT", "asset sale"),
    ],
)
def test_the_policys_excluded_credits_are_recognized(description, reason):
    """POL-DTI names these three kinds and they do not count towards income."""
    assert excluded_reason(description) == reason


@pytest.mark.parametrize(
    "description",
    [
        "SALARY  MONTHLY PAYROLL",
        "POS SETTLEMENT  CARD TERMINAL",
        "CUSTOMER PAYMENT  INVOICE 338",
    ],
)
def test_trading_credits_still_count_as_income(description):
    """Include credits that do not match exclusion patterns."""
    assert excluded_reason(description) is None


# ---------------------------------------------------------------------------
# Signs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cell", "value"),
    [
        ("1,200.00", 1200.0),
        ("-1,200.00", -1200.0),
        ("(1,200.00)", -1200.0),
        ("1,200.00-", -1200.0),
        ("  -56,000 ", -56000.0),
        ("", 0.0),
        ("-", 0.0),
        ("n/a", 0.0),
    ],
)
def test_an_amount_keeps_the_sign_the_statement_printed(cell, value):
    """A negative is written three ways and all three mean the same thing."""
    assert _money(cell) == value


@pytest.mark.parametrize("cell", ["about 500", "N0TANUMBER", "1.2.3"])
def test_an_amount_that_is_not_a_number_is_refused(cell):
    """Reject unrecognized amount strings."""
    with pytest.raises(Unreadable):
        _money(cell)


# ---------------------------------------------------------------------------
# Not checked is not the same as reconciled
# ---------------------------------------------------------------------------
def test_a_statement_with_nothing_in_it_is_refused():
    """Reject summaries with no extracted transactions."""
    with pytest.raises(Unreadable, match="no transactions"):
        validate(summary(transactions=0, reconciliation=NOT_CHECKED))


def test_a_statement_without_a_case_reference_is_refused():
    """Reject summaries without a case reference."""
    with pytest.raises(Unreadable, match="case reference"):
        validate(summary(case_id=""))


def test_a_single_row_is_reported_as_not_checked():
    """Report not_checked when only one transaction is available."""
    kept = validate(summary(transactions=1, reconciliation=NOT_CHECKED))
    assert kept.reconciliation == NOT_CHECKED
    assert kept.balance_breaks == []


# ---------------------------------------------------------------------------
# The diagnostics are read by the model and stored, so they are masked too
# ---------------------------------------------------------------------------
def test_a_balance_break_names_the_row_without_the_holders_details():
    """Mask personal details in reconciliation errors."""
    rows = [
        Transaction("2026-03-01", "SALARY  MONTHLY PAYROLL", 0, 85_000, 120_000),
        Transaction(
            "2026-03-03",
            "TRANSFER TO ELENA BRENNAN 4929123456789012 elena@example.com",
            3_400,
            0,
            82_600,
        ),
    ]
    breaks = reconcile(rows, "Elena Brennan")
    assert len(breaks) == 1
    assert breaks[0].startswith("row 2 on 2026-03-03")
    assert "ELENA" not in breaks[0].upper()
    assert "4929" not in breaks[0]
    assert "elena@example.com" not in breaks[0]
    assert "[HOLDER]" in breaks[0] and "[NUMBER]" in breaks[0]
