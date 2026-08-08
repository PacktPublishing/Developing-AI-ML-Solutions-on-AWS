"""Generate a synthetic SME underwriting-memo corpus for the knowledge base.

Two shapes mirror a real corpus without any real data: a structured 8-section
credit memo (the extractable one, with a Customer Financial Snapshot table) and
an Outlook approval thread (mostly boilerplate around a few lines of reasoning).
All names, businesses, amounts, and identifiers are invented; amounts are in USD.

Usage:
  python gen_memos.py --out data/generated/memos --count 200 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path

FIRST_NAMES = [
    "Marcus",
    "Elena",
    "Priya",
    "Devon",
    "Sofia",
    "Andre",
    "Naomi",
    "Terrence",
    "Claire",
    "Hassan",
    "Yuki",
    "Gabriel",
    "Rosa",
    "Malik",
    "Ingrid",
    "Cyrus",
    "Delia",
    "Omar",
    "Beatrice",
    "Nolan",
    "Farah",
    "Quentin",
    "Lucia",
    "Reggie",
]
LAST_NAMES = [
    "Delgado",
    "Okafor",
    "Whitfield",
    "Nakamura",
    "Brennan",
    "Castellano",
    "Abara",
    "Lindqvist",
    "Faulkner",
    "Mensah",
    "Petrov",
    "Sandoval",
    "Kowalski",
    "Ellison",
    "Rahman",
    "Guerrero",
    "Ashford",
    "Tomlin",
    "Vargas",
    "Bianchi",
]
CITIES = [
    ("Akron", "OH"),
    ("Fresno", "CA"),
    ("Tacoma", "WA"),
    ("Mobile", "AL"),
    ("Laredo", "TX"),
    ("Dayton", "OH"),
    ("Provo", "UT"),
    ("Peoria", "IL"),
    ("Augusta", "GA"),
    ("Reno", "NV"),
    ("Spokane", "WA"),
    ("Erie", "PA"),
]
SECTORS = [
    ("Grocery & Provisions", "wholesale grocery and dry goods", "Business Loan"),
    ("Logistics", "last-mile delivery and freight", "Working Capital"),
    ("Pharmacy", "retail pharmacy and medical supplies", "Business Loan"),
    ("Building Materials", "supply of cement, tiles, and hardware", "Asset Finance"),
    ("Textiles", "supply of fabrics and finished garments", "Business Loan"),
    ("Agro Supplies", "distribution of seed, feed, and fertilizer", "Working Capital"),
    ("Restaurant Supply", "wholesale food service equipment", "Business Loan"),
    ("Auto Parts", "supply of replacement parts and accessories", "Asset Finance"),
]
SUFFIXES = ["LLC", "Trading Co.", "Supply", "& Sons", "Enterprises", "Group"]
STAFF = [
    "Priya Anand",
    "Marcus Boyle",
    "Elena Ruiz",
    "Devon Clarke",
    "Sofia Marchetti",
    "Andre Whitfield",
    "Naomi Osei",
    "Terrence Kwan",
    "Claire Donovan",
]


@dataclass
class Memo:
    """One generated memo: its filename parts, header fields, and body text."""

    loan_id: int
    borrower: str
    doc: str
    doc_type: str
    uploaded: str
    body: str
    has_figures: bool


def _money(rng: random.Random, lo: int, hi: int) -> int:
    """Return a round-ish dollar amount in [lo, hi], snapped to thousands."""
    return rng.randint(lo // 1000, hi // 1000) * 1000


def _business_name(rng: random.Random) -> tuple[str, str]:
    """Return a synthetic business name and its promoter's full name."""
    promoter = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    stem = promoter.split()[1]
    sector_word = rng.choice(["Fresh", "Prime", "Metro", "Cedar", "Northgate", "Unity"])
    return f"{sector_word} {stem} {rng.choice(SUFFIXES)}", promoter


def _maybe_typo(rng: random.Random, text: str, messy: bool) -> str:
    """Drop the occasional vowel to mimic hand-typed memos when messy is set."""
    if not messy or rng.random() > 0.15:
        return text
    vowels = [
        i for i, c in enumerate(text) if c in "aeiou" and i not in (0, len(text) - 1)
    ]
    if not vowels:
        return text
    i = rng.choice(vowels)
    return text[:i] + text[i + 1 :]


def _structured(rng: random.Random, loan_id: int, messy: bool) -> Memo:
    """Render the 8-section structured credit memo, the extractable shape."""
    name, promoter = _business_name(rng)
    sector, line, facility = rng.choice(SECTORS)
    city, state = rng.choice(CITIES)
    accounts = rng.randint(1, 3)
    total_in = _money(rng, 480_000, 14_000_000)
    avg_in = total_in // 12
    amount = _money(rng, 50_000, 2_000_000)
    tenor = rng.choice([3, 4, 6, 9, 12])
    dti = "" if (messy and rng.random() < 0.35) else f"{rng.randint(8, 44)}%"
    repay = "N/A" if rng.random() < 0.5 else f"${_money(rng, 5_000, 60_000):,}/mo"
    rm, head = rng.sample(STAFF, 2)

    body = textwrap.dedent(f"""\
        SME Credit Memo - {name}

        1. Customer Information
        - Business Name: {_maybe_typo(rng, name, messy)}
        - Promoter(s) / Key Contact: {promoter}
        - Registration No.: {rng.randint(100000, 999999)}
        - Date of Incorporation: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/20{rng.randint(15, 24)}
        - Business Address: {rng.randint(10, 9999)} {rng.choice(["Market", "Depot", "Industrial", "Commerce"])} St, {city}, {state}
        - Sector / Line of Business: {sector}
        - Customer Type: {rng.choice(["new client", "existing client"])}

        2. Business Overview
        {_maybe_typo(rng, name, messy)} is a {sector.lower()} business in {city}, {state},
        engaged in {line}. The promoter has operated the business for
        {rng.randint(2, 15)} years and supplies wholesalers and retailers across the region.

        3. Loan Request Summary
        - Facility Type: {facility}
        - Purpose of Loan: Purchase of inventory worth ${amount:,}
        - Amount Requested: ${amount:,}
        - Tenor Requested: {tenor} months
        - Repayment Source: Proceeds from business sales

        4. Customer Financial Snapshot
        Metric                                Value
        Number of Bank accounts provided      {accounts}
        Total Inflows                         {total_in:,}
        Average Monthly Inflows               {avg_in:,}
        Existing Loan Repayments (if any)     {repay}
        Observed DTI %                        {dti}

        5. Credit History / Existing Exposure
        - History with lenders: {rng.choice(["No Credit History", "Satisfactory", "One prior facility, repaid"])}
        - Credit Bureau Status: {rng.choice(["Satisfactory", "No adverse records"])}

        6. Justification & Recommendation
        - Assessment: Stable cash flow and a consistent purchase pattern over
          {rng.randint(2, 15)} years support the request.
        Recommendation:
        "I recommend approval of ${amount:,} for {tenor} months to support the
        customer's inventory needs, based on trading activity, cash flow, and
        repayment behavior."

        7. Submitted By
        - Relationship Manager: {rm}
        - Head of SME/Sales: {head}
        - Date: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026

        8. Required/Supporting Documents:
        - 2 bank statements
        - Registration documents and letter of request
        - Valid ID and passport photograph
        - Completed guarantor's form
        """)
    doc = "SME_MEMO"
    return Memo(
        loan_id, name.upper(), doc, "pdf", "2026-04-13T14:53:23+0000", body, True
    )


def _email_thread(rng: random.Random, loan_id: int) -> Memo:
    """Render a nested Outlook approval thread, the low-signal shape."""
    name, _ = _business_name(rng)
    amount = _money(rng, 500_000, 20_000_000)
    approver, lead, rm, cc1, cc2 = rng.sample(STAFF, 5)
    body = textwrap.dedent(f"""\
        Subject: Re: ${amount // 1_000_000}M APPROVAL IFO {name.upper()}
        From: {approver} <{approver.split()[0].lower()}@example.com>
        Date: 2023-08-31 20:51:34+02:00

        Okay to process please.
        Thank you

            From: {lead} - Lead, Credit Policy and Assurance
            Sent: Thursday, August 31, 2023 7:50 PM
            To: {rm}; {approver}
            Cc: {cc1}; {cc2}; SME Underwriting
            Subject: RE: ${amount // 1_000_000}M APPROVAL IFO {name.upper()}

            Approved subject to standard risk checks and all securities and
            guarantees being in place. Repayment instruments set up and confirmed
            by Risk before disbursement.

                From: {rm}
                Sent: Thursday, 31 August 2023 7:41 PM
                To: {approver}

                Hello, below mail refers. Kindly approve the business loan request
                for ${amount:,} for further processing.

                    From: {cc1}
                    Sent: Thursday, 31 August 2023 6:19 PM
                    [thread continues below with prior correspondence]
        """)
    doc = f"Re_{amount // 1_000_000}M_APPROVAL.msg"
    return Memo(
        loan_id, name.upper(), doc, "msg", "2023-08-31T18:54:59+0000", body, False
    )


def header(m: Memo) -> str:
    """Return the 4-line file header prepended to every memo body."""
    return (
        f"# loan_id: {m.loan_id}\n"
        f"# borrower: {m.borrower}\n"
        f"# document: {m.doc}\n"
        f"# type: {m.doc_type}   uploaded: {m.uploaded}\n" + "-" * 72 + "\n\n"
    )


def generate(out: Path, count: int, seed: int, messy: bool) -> list[Memo]:
    """Write `count` synthetic memos to `out` and return their metadata."""
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    memos: list[Memo] = []
    for i in range(count):
        loan_id = 34_600_000 + i
        # roughly two-thirds structured (extractable), one-third email threads
        m = (
            _structured(rng, loan_id, messy)
            if rng.random() < 0.65
            else _email_thread(rng, loan_id)
        )
        fname = f"{m.loan_id}__{m.borrower.replace(' ', '_')}__{m.doc}.txt"
        (out / fname).write_text(header(m) + m.body, encoding="utf-8")
        memos.append(m)
    return memos


def write_index(memos: list[Memo], path: Path) -> None:
    """Write the index CSV mapping each memo to its type, size, and figure flag."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "loan_id",
                "borrower",
                "document",
                "type",
                "char_count",
                "has_extracted_figures",
            ]
        )
        for m in memos:
            w.writerow(
                [
                    m.loan_id,
                    m.borrower,
                    m.doc,
                    m.doc_type,
                    len(m.body),
                    int(m.has_figures),
                ]
            )


def main() -> None:
    """Parse arguments and generate the corpus plus its index."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/generated/memos"))
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--messy", action="store_true", help="inject typos and blank fields")
    a = p.parse_args()
    memos = generate(a.out, a.count, a.seed, a.messy)
    write_index(memos, a.out.parent / "memo_index.csv")
    figures = sum(m.has_figures for m in memos)
    print(f"wrote {len(memos)} memos to {a.out} ({figures} with extractable figures)")


if __name__ == "__main__":
    main()
