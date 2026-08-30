"""Generate a synthetic SME underwriting-memo corpus for the knowledge base.

Three shapes mirror a real corpus without any real data: a structured 8-section
credit memo (the extractable one, with a Customer Financial Snapshot table), an
Outlook approval thread (mostly boilerplate around a few lines of reasoning), and
an amendment request against a facility that already exists. The third is the
common case in a real archive, where most decisions are servicing changes rather
than new lending: deferred principal, facility increases, security instruments,
tenor and repayment changes, and approvals above a limit.

All names, businesses, amounts, and identifiers are invented, and amounts are in
USD.

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


# The amendment types an archive actually fills up with, each with what is being
# asked for and the condition an approver usually attaches to it.
AMENDMENTS = [
    (
        "DEFERRED PRINCIPAL",
        "a {months}-month principal moratorium",
        "Interest continues to accrue and is collected monthly.",
    ),
    (
        "FACILITY INCREASE",
        "an increase of ${amount:,} on the existing facility",
        "Combined exposure stays inside the single-obligor limit.",
    ),
    (
        "DEFERRED GUARANTOR CHEQUE",
        "deferral of the guarantor cheque presentation by {months} month(s)",
        "Guarantor forms remain on file and unchanged.",
    ),
    (
        "CHANGE OF REPAYMENT ACCOUNT",
        "a change of the repayment account to another bank",
        "New direct debit mandate confirmed by Risk before the next cycle.",
    ),
    (
        "TENOR EXTENSION",
        "an extension of tenor from {tenor} to {new_tenor} months",
        "Instalment recalculated, and the rate is unchanged.",
    ),
    (
        "EXCEPTIONAL APPROVAL",
        "an exceptional approval of ${amount:,}, above the standard limit",
        "Approval is one-off and does not set a precedent for the sector.",
    ),
    (
        "RATE CONCESSION",
        "a rate concession to {rate}% monthly",
        "Concession is reviewed at the next annual credit review.",
    ),
    (
        "APPROVAL TO PROCEED",
        "approval to proceed with the application",
        "Subject to securities and repayment instruments being in place.",
    ),
]


def _amendment(rng: random.Random, loan_id: int, as_thread: bool) -> Memo:
    """Render a servicing decision on a facility that already exists.

    Most of a real archive is this: not the original credit assessment, but the
    changes made to it afterwards, each one a short request and a shorter answer.
    """
    name, _ = _business_name(rng)
    sector, _, facility = rng.choice(SECTORS)
    approver, lead, rm = rng.sample(STAFF, 3)
    kind, asked, condition = rng.choice(AMENDMENTS)
    months = rng.choice([1, 2, 3])
    tenor = rng.choice([6, 9, 12])
    fields = {
        "months": months,
        "amount": _money(rng, 500_000, 18_000_000),
        "tenor": tenor,
        "new_tenor": tenor + rng.choice([3, 6]),
        "rate": round(rng.uniform(3.5, 5.5), 1),
    }
    request = asked.format(**fields)
    exposure = _money(rng, 800_000, 20_000_000)
    dpd = rng.choice([0, 0, 0, rng.randint(1, 29)])
    reason = rng.choice(
        [
            "Receivables from two distributors now fall due a month later than planned.",
            "A large order was delivered but payment terms were renegotiated to 60 days.",
            "Seasonal demand moved the sales cycle out by several weeks.",
            "The promoter reinvested collections into stock ahead of a supply increase.",
            "A key customer settled late, and the shortfall is temporary.",
        ]
    )

    if as_thread:
        body = textwrap.dedent(f"""\
            Subject: RE: {kind} - {name.upper()}
            From: {approver} <{approver.split()[0].lower()}@example.com>
            Date: 2026-03-04 09:55:12+01:00

            Approved.

                From: {lead} - Credit Risk
                Sent: Monday, 2 March 2026 4:11 PM
                To: {rm}; {approver}
                Subject: RE: {kind} - {name.upper()}

                No objection from Credit Policy. {condition}

                    From: {rm}
                    Sent: Monday, 2 March 2026 2:11 PM
                    To: {lead}

                    Kindly approve {request} for the customer above.
                    {reason}

                    Exposure: ${exposure:,}   Tenor: {tenor} months
                    Days past due: {dpd}      Sector: {sector}
                    Collateral: guarantor cheques, 2 signatories
            """)
        doc = f"{kind.title().replace(' ', '_')}.msg"
        return Memo(
            loan_id, name.upper(), doc, "msg", "2026-03-04T08:55:12+0000", body, False
        )

    body = textwrap.dedent(f"""\
        Facility Amendment Request - {name}

        1. Request
        - Type: {kind.title()}
        - Facility: {facility}
        - Requesting: {request}

        2. Current Position
        Metric                                Value
        Outstanding Exposure                  {exposure:,}
        Original Tenor                        {tenor} months
        Days Past Due                         {dpd}
        Credit Bureau Status                  {rng.choice(["Satisfactory", "No adverse records"])}
        Collateral                            Guarantor cheques, 2 signatories

        3. Background
        {reason} The promoter has serviced the facility without
        arrears prior to this request.

        4. Recommendation
        Recommendation:
        "I recommend approval of {request}. {condition}"

        - Relationship Manager: {rm}
        - Credit Policy: {lead}
        - Date: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026
        """)
    doc = f"{kind.title().replace(' ', '_')}_Request"
    return Memo(
        loan_id, name.upper(), doc, "pdf", "2026-03-04T08:55:12+0000", body, True
    )


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
        Subject: Re: ${amount // 1_000_000}M APPROVAL - {name.upper()}
        From: {approver} <{approver.split()[0].lower()}@example.com>
        Date: 2023-08-31 20:51:34+02:00

        Okay to process please.
        Thank you

            From: {lead} - Credit Risk
            Sent: Thursday, August 31, 2023 7:50 PM
            To: {rm}; {approver}
            Cc: {cc1}; {cc2}; SME Underwriting
            Subject: RE: ${amount // 1_000_000}M APPROVAL - {name.upper()}

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
        # The mix follows a real archive rather than an even split. Amendments to
        # existing facilities dominate, and about half of everything arrives as a
        # mail trail rather than a form.
        roll = rng.random()
        if roll < 0.45:
            m = _amendment(rng, loan_id, as_thread=rng.random() < 0.55)
        elif roll < 0.80:
            m = _structured(rng, loan_id, messy)
        else:
            m = _email_thread(rng, loan_id)
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
