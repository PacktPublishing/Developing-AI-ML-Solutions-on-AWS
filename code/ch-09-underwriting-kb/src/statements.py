"""Read an applicant's bank statement PDF and total it.

Extracting the transactions is parsing, and totalling them is arithmetic, so
both belong here rather than in a prompt: the agent reads the figures from this
module the same way it reads a debt-to-income ratio from the affordability tool.

Usage:
  PYTHONPATH=src uv run src/statements.py data/generated/statements/CASE-*.pdf
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pdfplumber

# a debit is an existing obligation when its narrative names a facility
OBLIGATION = re.compile(r"loan|repayment|facility|instal?ment", re.IGNORECASE)
HEADER = ("date", "description", "money out", "money in", "balance")
# card and account numbers run to eight digits or more; a facility reference does not
LONG_NUMBER = re.compile(r"\b\d(?:[ -]?\d){7,}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


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
    monthly_income: float
    existing_repayments: float
    transactions: int
    # rows whose balance does not follow from the one before: a misread line
    balance_breaks: list[str]


def _money(cell: str | None) -> float:
    """Return a cell as a number, treating blanks and stray marks as zero."""
    if not cell:
        return 0.0
    cleaned = re.sub(r"[^0-9.]", "", cell)
    return float(cleaned) if cleaned else 0.0


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


def reconcile(rows: list[Transaction]) -> list[str]:
    """Return each row whose balance is not the previous balance plus in, minus out.

    The statement prints no opening balance, so the first row is the anchor and
    every later row is checked against the one before it.
    """
    breaks = []
    for prev, t in pairwise(rows):
        expected = prev.balance + t.inn - t.out
        if abs(t.balance - expected) > 0.5:
            breaks.append(
                f"{t.date} {t.description}: balance {t.balance:,.0f},"
                f" expected {expected:,.0f}"
            )
    return breaks


def summarise(path: str | Path) -> Summary:
    """Return the monthly income and existing repayments the statement supports."""
    fields, rows = read(path)
    by_month_in: dict[str, float] = defaultdict(float)
    by_month_obligation: dict[str, float] = defaultdict(float)
    for t in rows:
        month = t.date[:7]
        by_month_in[month] += t.inn
        if t.out and OBLIGATION.search(t.description):
            by_month_obligation[month] += t.out
    months = len(by_month_in) or 1
    return Summary(
        case_id=fields.get("case reference", ""),
        holder=fields.get("account holder", ""),
        months=months,
        monthly_income=round(sum(by_month_in.values()) / months, 2),
        existing_repayments=round(sum(by_month_obligation.values()) / months, 2),
        transactions=len(rows),
        balance_breaks=reconcile(rows),
    )


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
    store.index_submission(
        runtime,
        summary.case_id,
        summary.case_id,
        chunks,
        meta={
            "monthly_income": summary.monthly_income,
            "existing_repayments": summary.existing_repayments,
            "months": summary.months,
            "balance_breaks": summary.balance_breaks,
        },
    )
    return summary


def main() -> None:
    """Print the summary for each statement given on the command line."""
    for arg in sys.argv[1:]:
        s = summarise(arg)
        print(
            f"{s.case_id}  {s.holder}: income {s.monthly_income:,.0f}/month, "
            f"existing repayments {s.existing_repayments:,.0f}/month "
            f"over {s.months} months, {s.transactions} transactions"
        )


if __name__ == "__main__":
    main()
