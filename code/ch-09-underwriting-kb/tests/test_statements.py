"""The statement parser must recover the figures the generator wrote."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("reportlab")

from statements import Transaction, chunks_for, mask, reconcile, summarise

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Generate a small statement set and return its directory and manifest."""
    out = tmp_path_factory.mktemp("statements")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "etl" / "gen_statements.py"),
            "--out",
            str(out),
            "--count",
            "6",
            "--seed",
            "3",
        ],
        check=True,
        capture_output=True,
    )
    rows = list(csv.DictReader((out / "manifest.csv").open()))
    return out, rows


def test_parser_recovers_the_generated_figures(corpus):
    """Income and existing repayments must match the manifest exactly."""
    out, rows = corpus
    for row in rows:
        s = summarise(out / row["file"])
        assert s.case_id == row["case_id"]
        assert s.monthly_income == pytest.approx(float(row["monthly_income"]), abs=1)
        assert s.existing_repayments == pytest.approx(
            float(row["existing_repayment"]), abs=1
        )


def test_applicants_without_a_facility_show_no_repayments(corpus):
    """A statement with no loan debit must total to zero, not to something small."""
    out, rows = corpus
    clean = [r for r in rows if float(r["existing_repayment"]) == 0]
    if not clean:
        pytest.skip("this seed gave every applicant a facility")
    for row in clean:
        assert summarise(out / row["file"]).existing_repayments == 0


def test_chunks_lead_with_the_figures_and_group_by_month(corpus):
    """The first chunk states the figures; the rest are one month each."""
    out, rows = corpus
    summary, chunks = chunks_for(out / rows[0]["file"])
    assert str(int(summary.monthly_income)) in chunks[0].replace(",", "")
    assert len(chunks) == summary.months + 1
    for chunk in chunks[1:]:
        assert chunk.startswith("Transactions for ")


def test_chunks_carry_the_case_not_the_holder(corpus):
    """What is embedded names the case; the holder's name stays out of it."""
    out, rows = corpus
    summary, chunks = chunks_for(out / rows[0]["file"])
    text = "\n".join(chunks)
    assert summary.case_id in chunks[0]
    for part in summary.holder.split():
        assert part not in text


def test_mask_keeps_what_the_affordability_check_reads():
    """Names, card numbers, and emails go; the facility and amounts stay."""
    line = mask(
        "TRANSFER FROM ELENA BRENNAN 4929 1234 5678 9012 elena@example.com",
        "Elena Brennan",
    )
    assert line == "TRANSFER FROM [HOLDER] [NUMBER] [EMAIL]"
    kept = mask("LOAN REPAYMENT  FACILITY 4471", "Elena Brennan")
    assert kept == "LOAN REPAYMENT  FACILITY 4471"


def test_generated_statements_reconcile(corpus):
    """Every row of a clean statement follows from the one before it."""
    out, rows = corpus
    for row in rows:
        assert summarise(out / row["file"]).balance_breaks == []


def test_a_misread_amount_breaks_the_running_balance():
    """A dropped digit shows up as the row whose balance no longer follows."""
    rows = [
        Transaction("2026-03-01", "SALARY", 0, 85_000, 120_000),
        Transaction("2026-03-03", "LOAN REPAYMENT", 3_400, 0, 82_600),
        Transaction("2026-03-06", "FUEL STATION", 100, 0, 82_500),
    ]
    breaks = reconcile(rows)
    assert len(breaks) == 1
    assert breaks[0].startswith("2026-03-03 LOAN REPAYMENT")
