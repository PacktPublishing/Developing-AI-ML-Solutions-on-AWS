# /// script
# dependencies = []
# ///
"""Where each team's work begins and ends, written once for both routers.

The zero-shot LLM is handed these as a prompt, a line per team. The decision
model is handed the same text as the `criteria` of a choice question, which is
where its options are described. One definition serves both, so a comparison
between the two routers is a comparison of the routers rather than of two
differently worded briefs.

Two tie-break rules sit beside the definitions because the broad-sounding teams
collect what belongs to the specific ones: a request to borrow money reads as
savings to a model that has only seen the word "deposit", and a login failure
reads as whatever product the customer was trying to reach.
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
