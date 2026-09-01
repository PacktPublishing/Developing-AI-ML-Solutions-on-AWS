"""The statement parser must recover the figures the generator wrote."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("reportlab")

from statements import chunks_for, summarise

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
