# /// script
# dependencies = ["reportlab"]
# ///
"""Generate synthetic bank statement PDFs for applicants awaiting a decision.

Each statement is three months of transactions for one applicant: a recurring
salary credit, the repayments on any facility they already hold, and everyday
spending around them. The manifest records the true monthly income and existing
repayments for each file, so the parser can be checked against them rather than
against a number somebody eyeballed.

All names, businesses, amounts, and account numbers are invented, and amounts
are in USD.

Usage:
  python gen_statements.py --out data/generated/statements --count 12 --seed 11
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BANK = "Meridian Commercial Bank"
FIRST = [
    "Marcus",
    "Elena",
    "Priya",
    "Tobias",
    "Ingrid",
    "Rafael",
    "Yuki",
    "Amara",
    "Soren",
    "Lucia",
    "Dmitri",
    "Farah",
    "Callum",
    "Neve",
    "Anders",
]
LAST = [
    "Whitfield",
    "Castellano",
    "Lindqvist",
    "Sandoval",
    "Ashford",
    "Brennan",
    "Guerrero",
    "Nakamura",
    "Ellison",
    "Bianchi",
    "Marchetti",
    "Tomlin",
]
EVERYDAY = [
    ("POS PURCHASE  GROCERY MARKET", 120, 900),
    ("POS PURCHASE  FUEL STATION", 80, 420),
    ("TRANSFER TO  SAVINGS", 200, 1500),
    ("DIRECT DEBIT  UTILITIES", 90, 320),
    ("POS PURCHASE  PHARMACY", 40, 260),
    ("ATM WITHDRAWAL", 100, 600),
    ("DIRECT DEBIT  MOBILE AIRTIME", 25, 90),
    ("POS PURCHASE  HARDWARE SUPPLIES", 150, 1100),
]


@dataclass
class Statement:
    """One applicant's statement and the figures a parser should recover."""

    case_id: str
    holder: str
    account: str
    monthly_income: int
    existing_repayment: int
    rows: list[tuple[str, str, str, str, str]]


def build(rng: random.Random, case_id: str) -> Statement:
    """Compose three months of transactions around a salary and any repayment."""
    holder = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    account = f"****{rng.randint(1000, 9999)}"
    income = rng.randrange(45_000, 260_000, 5_000)
    # about half the applicants already carry a facility somewhere
    repayment = rng.randrange(4_000, 40_000, 1_000) if rng.random() < 0.55 else 0
    balance = rng.randrange(20_000, 400_000, 1_000)

    rows: list[tuple[str, str, str, str, str]] = []
    start = dt.date(2026, 3, 1)
    for month in range(3):
        first = (start.replace(day=1) + dt.timedelta(days=32 * month)).replace(day=1)
        entries: list[tuple[int, str, int, int]] = [
            (26, "SALARY  MONTHLY PAYROLL", 0, income)
        ]
        if repayment:
            entries.append((3, "LOAN REPAYMENT  FACILITY 4471", repayment, 0))
        for _ in range(rng.randint(5, 9)):
            label, low, high = rng.choice(EVERYDAY)
            entries.append((rng.randint(1, 28), label, rng.randrange(low, high, 10), 0))
        for day, label, out, inn in sorted(entries):
            balance += inn - out
            rows.append(
                (
                    (first + dt.timedelta(days=day - 1)).isoformat(),
                    label,
                    f"{out:,}" if out else "",
                    f"{inn:,}" if inn else "",
                    f"{balance:,}",
                )
            )
    return Statement(case_id, holder, account, income, repayment, rows)


def write_pdf(st: Statement, path: Path) -> None:
    """Render one statement as a PDF with a header block and a transaction table."""
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Statement {st.case_id}",
    )
    head = [
        f"<b>{BANK}</b>",
        "Statement of account",
        f"Account holder: {st.holder}",
        f"Account number: {st.account}",
        f"Case reference: {st.case_id}",
        "Period: 01 March 2026 to 31 May 2026",
    ]
    flow = [Paragraph(line, styles["Normal"]) for line in head]
    flow.append(Spacer(1, 8 * mm))
    data = [["Date", "Description", "Money out", "Money in", "Balance"]] + [
        list(r) for r in st.rows
    ]
    table = Table(
        data, colWidths=[24 * mm, 66 * mm, 26 * mm, 26 * mm, 28 * mm], repeatRows=1
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECF1")),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8C0CC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    flow.append(table)
    doc.build(flow)


def main() -> None:
    """Write the statement PDFs and a manifest of their true figures."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/generated/statements")
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pdf"):
        old.unlink()

    rng = random.Random(args.seed)
    manifest = []
    for i in range(args.count):
        st = build(rng, f"CASE-{2026}{i:04d}")
        name = f"{st.case_id}__{st.holder.replace(' ', '_')}__statement.pdf"
        write_pdf(st, out / name)
        manifest.append(
            {
                "case_id": st.case_id,
                "file": name,
                "holder": st.holder,
                "monthly_income": st.monthly_income,
                "existing_repayment": st.existing_repayment,
            }
        )
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"wrote {len(manifest)} statements to {out}")


if __name__ == "__main__":
    main()
