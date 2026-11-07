# /// script
# dependencies = []
# ///
"""Define categories shared by the LLM and decision-model routers.

Use the definitions in LLM prompts and decision-model choice criteria. Include
tie-break rules for borrowing requests and digital-banking issues.
"""

# Ordered as the label file is, so a routing table reads in the same order
# wherever it is printed.
BORDERS: dict[str, str] = {
    "Bank Accounts": "current/checking accounts, statements, fees, balances",
    "Cards": "debit/credit cards, activation, limits, lost or stolen, card fraud",
    "Customer Rewards": "points, cashback, loyalty, referrals",
    "Digital Banking": (
        "the mobile app or online banking, login, passwords, transfers, errors"
    ),
    "Insurance": "policies and claims (life, home, travel, health)",
    "Investment Funds": "mutual funds, ETFs, brokerage, buying/selling securities",
    "Mortgages": "home/property loans, rates, refinancing",
    "Pension Plans": "retirement/pension contributions and withdrawals",
    "Personal Loans": "unsecured consumer loans and their repayment (not property)",
    "Savings & Deposits": (
        "savings accounts and fixed/term deposits and their interest ONLY"
    ),
}

RULES = (
    "Rules: borrowing money is Personal Loans or Mortgages, never Savings &"
    " Deposits; app/login/transfer issues are Digital Banking."
)


def as_lines() -> str:
    """Return the definitions as the lines a prompt lists them on."""
    return "\n".join(f"- {team}: {border}." for team, border in BORDERS.items())


def as_criteria(labels: list[str], described: bool) -> dict[str, str | None]:
    """Return the `criteria` of a choice question for the given labels.

    A decision model always scores named options, so the question carries the
    names either way. `described` is what the chapter varies: None leaves an
    option as a bare label, and a definition draws its border, which is the same
    difference the two prompt variants make to the LLM.
    """
    if not described:
        return dict.fromkeys(labels, None)
    return {label: BORDERS.get(label) for label in labels}
