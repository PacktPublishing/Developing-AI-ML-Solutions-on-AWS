"""Read an applicant's bank statement PDF and total it.

Extracting the transactions is parsing, and totalling them is arithmetic, so
both belong here rather than in a prompt: the agent reads the figures from this
module the same way it reads a debt-to-income ratio from the affordability tool.

Two distinctions the credit policy makes and a parser has to make with it.
Not every credit is income: POL-DTI excludes transfers between the borrower's
own accounts, one-off asset sales, and loan disbursements, so the total of the
credits and the income the ratio may use are different numbers and both are
reported. And a statement nothing could be read from is not a reconciled
statement, so an extraction that recovered no transactions is refused rather
than passed on as zero income.

Usage:
  PYTHONPATH=src uv run src/statements.py data/generated/statements/CASE-*.pdf
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import pdfplumber

# a debit is an existing obligation when its narrative names a facility
OBLIGATION = re.compile(r"loan|repayment|facility|instal?ment", re.IGNORECASE)
HEADER = ("date", "description", "money out", "money in", "balance")
# card and account numbers run to eight digits or more; a facility reference does not
LONG_NUMBER = re.compile(r"\b\d(?:[ -]?\d){7,}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Credits POL-DTI does not count as income: money the borrower moved between
# their own accounts, the proceeds of a one-off sale, and a facility paying out.
# Each pattern carries the reason, so an excluded credit can say why it was
# excluded rather than quietly leaving the total.
NOT_INCOME: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "own-account transfer",
        re.compile(
            r"transfer\s+(?:from|in)\b.*\b(?:own|self|savings|current)\b|own account",
            re.IGNORECASE,
        ),
    ),
    (
        "loan disbursement",
        re.compile(
            r"disburse?ment|disbursed|drawdown|draw down|facility advance",
            re.IGNORECASE,
        ),
    ),
    (
        "asset sale",
        re.compile(r"asset sale|sale of\b|proceeds of sale", re.IGNORECASE),
    ),
)

# cells that mean "nothing here" rather than a number
BLANK_CELLS = {"", "-", "--", "nil", "none", "n/a", "na"}

RECONCILED = "reconciled"
BREAKS = "breaks"
NOT_CHECKED = "not_checked"


class Unreadable(ValueError):
    """Raised when a PDF yielded nothing a review could be based on."""


@dataclass
class Transaction:
    """One line of the statement."""

    date: str
    description: str
    out: float
    inn: float
    balance: float


@dataclass
class Summary:
    """What the statement says about capacity, for the affordability tools."""

    case_id: str
    holder: str
    months: int
    # every credit on the statement, averaged over the months it covers
    total_inflows: float
    # the part of that the policy lets a ratio use: total less the excluded kinds
    monthly_income: float
    existing_repayments: float
    transactions: int
    # whether the running balance was checked at all, and what the check said
    reconciliation: str
    # rows whose balance does not follow from the one before: a misread line
    balance_breaks: list[str]
    # credits left out of income, masked, with the reason each was left out
    excluded_inflows: list[str] = field(default_factory=list)
    # which upload of this case the figures belong to, set when it is indexed
    version: int = 0


def _money(cell: str | None) -> float:
    """Return a cell as a signed number, or raise when it is not one.

    Statements write a negative either way round, as -1,200.00 and as
    (1,200.00), and some print the sign after the digits. All three mean the
    same thing, and an overdrawn balance read as a positive one turns a
    reversal into a receipt, so the sign is kept rather than stripped.

    What is left after the decorations come off has to be a number. Reducing a
    cell to the digits it happens to contain is what turns "about 500" into 500
    and a column of initials into zeroes, so anything else is refused and the
    statement is refused with it.
    """
    if cell is None:
        return 0.0
    text = cell.strip()
    if text.lower() in BLANK_CELLS:
        return 0.0
    negative = (
        text.startswith("-")
        or text.endswith("-")
        or (text.startswith("(") and text.endswith(")"))
    )
    core = text.strip("()").strip().lstrip("-").rstrip("-").strip()
    core = core.lstrip("$\u20ac\u00a3\u20a6").strip().replace(",", "").replace(" ", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", core):
        raise Unreadable(f"unrecognised amount {cell!r}")
    value = float(core)
    return -value if negative else value


def excluded_reason(description: str) -> str | None:
    """Return why this credit is not income, or None when it counts."""
    for reason, pattern in NOT_INCOME:
        if pattern.search(description):
            return reason
    return None


def read(path: str | Path) -> tuple[dict[str, str], list[Transaction]]:
    """Return the statement's header fields and its transactions."""
    fields: dict[str, str] = {}
    rows: list[Transaction] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip().lower()
                    if key in ("account holder", "case reference", "account number"):
                        fields[key] = value.strip()
            for table in page.extract_tables():
                for row in table:
                    if not row or len(row) < 5:
                        continue
                    if (row[0] or "").strip().lower() == "date":
                        continue
                    date = (row[0] or "").strip()
                    if not re.match(r"\d{4}-\d{2}-\d{2}", date):
                        continue
                    rows.append(
                        Transaction(
                            date,
                            (row[1] or "").strip(),
                            _money(row[2]),
                            _money(row[3]),
                            _money(row[4]),
                        )
                    )
    return fields, rows


def reconcile(rows: list[Transaction], holder: str = "") -> list[str]:
    """Return each row whose balance is not the previous balance plus in, minus out.

    The statement prints no opening balance, so the first row is the anchor and
    every later row is checked against the one before it. A break names the row
    by its number and date, and its narrative is masked the way the indexed text
    is: the diagnostic is handed to the model and stored with the submission, so
    it cannot be the one place the holder's details travel in the clear.
    """
    breaks = []
    for position, (prev, t) in enumerate(pairwise(rows), start=2):
        expected = prev.balance + t.inn - t.out
        if abs(t.balance - expected) > 0.5:
            breaks.append(
                f"row {position} on {t.date} ({mask(t.description, holder)}):"
                f" balance {t.balance:,.0f}, expected {expected:,.0f}"
            )
    return breaks


def summarise(path: str | Path) -> Summary:
    """Return the monthly income and existing repayments the statement supports."""
    fields, rows = read(path)
    holder = fields.get("account holder", "")
    by_month_total: dict[str, float] = defaultdict(float)
    by_month_income: dict[str, float] = defaultdict(float)
    by_month_obligation: dict[str, float] = defaultdict(float)
    excluded: list[str] = []
    for t in rows:
        month = t.date[:7]
        if t.inn:
            by_month_total[month] += t.inn
            reason = excluded_reason(t.description)
            if reason:
                excluded.append(
                    f"{t.date} {mask(t.description, holder)} {t.inn:,.0f}: {reason}"
                )
            else:
                by_month_income[month] += t.inn
        if t.out and OBLIGATION.search(t.description):
            by_month_obligation[month] += t.out
    months = len({t.date[:7] for t in rows}) or 1
    breaks = reconcile(rows, holder)
    return Summary(
        case_id=fields.get("case reference", ""),
        holder=holder,
        months=months,
        total_inflows=round(sum(by_month_total.values()) / months, 2),
        monthly_income=round(sum(by_month_income.values()) / months, 2),
        existing_repayments=round(sum(by_month_obligation.values()) / months, 2),
        transactions=len(rows),
        # one row cannot disagree with the row before it, so there is nothing to
        # check and nothing to report as checked
        reconciliation=(
            NOT_CHECKED if len(rows) < 2 else (BREAKS if breaks else RECONCILED)
        ),
        balance_breaks=breaks,
        excluded_inflows=excluded,
    )


def validate(summary: Summary) -> Summary:
    """Return the summary, or refuse it when there is nothing to review.

    A PDF whose headers parsed and whose table did not gives zero income, zero
    repayments and an empty list of balance breaks, which reads downstream as a
    reconciled statement from an applicant with no income. Refuse it here, at
    the point the figures are produced, rather than indexing it and leaving the
    agent to tell the difference.
    """
    if not summary.transactions:
        raise Unreadable(
            "no transactions recognised: the statement may be a scan, or a layout"
            " the table extractor does not read"
        )
    if not summary.case_id:
        raise Unreadable("no case reference in the statement header")
    return summary


def mask(text: str, holder: str) -> str:
    """Replace the holder's name, long numbers, and email addresses in one line.

    The name comes from the statement's own header, so it is matched exactly
    rather than guessed at by a model.
    """
    # an address can contain the name, so it goes first
    text = LONG_NUMBER.sub("[NUMBER]", EMAIL.sub("[EMAIL]", text))
    for part in [holder, *holder.split()]:
        if len(part) > 2:
            text = re.sub(
                rf"\b{re.escape(part)}\b", "[HOLDER]", text, flags=re.IGNORECASE
            )
    return text


def chunks_for(path: str | Path) -> tuple[Summary, list[str]]:
    """Return the summary and the statement rendered as one chunk per month.

    A statement is a table, so it is grouped by month rather than split on
    paragraph breaks: each chunk is a period a reader would actually compare.
    The chunks are masked, because they are embedded and stored; the case id
    is what ties them back to the applicant.
    """
    _, rows = read(path)
    summary = summarise(path)
    header = (
        f"Bank statement for case {summary.case_id}. "
        f"Monthly income {summary.monthly_income:,.0f}. "
        f"Total credits {summary.total_inflows:,.0f} a month. "
        f"Existing repayments {summary.existing_repayments:,.0f} a month."
    )
    by_month: dict[str, list[Transaction]] = defaultdict(list)
    for t in rows:
        by_month[t.date[:7]].append(t)
    out = [header]
    for month in sorted(by_month):
        lines = [f"Transactions for {month}:"]
        for t in by_month[month]:
            money = f"out {t.out:,.0f}" if t.out else f"in {t.inn:,.0f}"
            lines.append(f"{t.date}  {mask(t.description, summary.holder)}  {money}")
        out.append("\n".join(lines))
    return summary, out


def submit(path: str | Path, store, runtime) -> Summary:
    """Parse an uploaded statement and index it as the document under review."""
    summary, chunks = chunks_for(path)
    validate(summary)
    summary.version = store.index_submission(
        runtime,
        summary.case_id,
        summary.case_id,
        chunks,
        meta={
            "monthly_income": summary.monthly_income,
            "total_inflows": summary.total_inflows,
            "existing_repayments": summary.existing_repayments,
            "months": summary.months,
            "reconciliation": summary.reconciliation,
            "balance_breaks": summary.balance_breaks,
            "excluded_inflows": summary.excluded_inflows,
        },
    )
    return summary


def main() -> None:
    """Print the summary for each statement given on the command line."""
    for arg in sys.argv[1:]:
        try:
            s = validate(summarise(arg))
        except Unreadable as err:
            print(f"{arg}: refused, {err}")
            continue
        print(
            f"{s.case_id}  {s.holder}: income {s.monthly_income:,.0f}/month "
            f"of {s.total_inflows:,.0f} credits, existing repayments "
            f"{s.existing_repayments:,.0f}/month over {s.months} months, "
            f"{s.transactions} transactions, {s.reconciliation}"
        )


if __name__ == "__main__":
    main()
