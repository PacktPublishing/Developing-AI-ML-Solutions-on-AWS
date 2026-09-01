# /// script
# dependencies = ["boto3", "ollama", "opensearch-py", "pdfplumber", "strands-agents"]
# ///
"""Review one uploaded bank statement against the archive of past decisions.

Uploads the statement into OpenSearch, then asks the agent for a review: the
statement supplies the figures, the affordability tools do the arithmetic, and
the memo archive supplies the precedent, cited by loan id.

Usage:
  PYTHONPATH=src uv run src/review.py --case CASE-20260000 --tenor 12 --rate 18
"""

import argparse
from pathlib import Path

from agent import build_agent
from models import get_runtime
from statements import submit
from stores import get_store

STATEMENTS = Path("data/generated/statements")


def find(case_id: str) -> Path:
    """Return the statement PDF for a case reference."""
    matches = sorted(STATEMENTS.glob(f"{case_id}__*.pdf"))
    if not matches:
        raise SystemExit(
            f"no statement for {case_id} in {STATEMENTS}; run make statements"
        )
    return matches[0]


def main() -> None:
    """Upload one statement and print the agent's review of it."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case", required=True, help="the case reference, e.g. CASE-20260000"
    )
    p.add_argument("--tenor", type=int, default=12, help="tenor in months")
    p.add_argument("--rate", type=float, default=18.0, help="annual rate percent")
    p.add_argument("--keep", action="store_true", help="leave the statement indexed")
    args = p.parse_args()

    store, runtime = get_store(), get_runtime()
    summary = submit(find(args.case), store, runtime)
    print(
        f"uploaded {summary.case_id} ({summary.holder}): "
        f"{summary.transactions} transactions over {summary.months} months\n"
    )
    try:
        print(
            build_agent()(
                f"Review {args.case} for a {args.tenor} month facility at "
                f"{args.rate} percent. Read the statement, work out what the "
                f"numbers allow, and say whether we have lent on similar terms before."
            )
        )
    finally:
        if not args.keep:
            store.drop_submission(args.case)


if __name__ == "__main__":
    main()
