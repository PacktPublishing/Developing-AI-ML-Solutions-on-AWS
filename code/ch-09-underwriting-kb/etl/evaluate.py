# /// script
# dependencies = ["boto3", "ollama", "opensearch-py"]
# ///
"""Check retrieval against questions whose right answer is known.

Three kinds of question, each built from the corpus itself: one naming a
borrower, one naming a loan id, and one about credit policy. A question counts as
found when a right document is among the top k. Each runs twice, k-NN alone and
hybrid, so a change to chunking, weights, or the model shows up as a number.

Usage (from the chapter root, with the index seeded):
  PYTHONPATH=src uv run etl/evaluate.py --memos data/generated/memos
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from embed_memos import parse_memo
from models import get_runtime
from stores import get_store

POLICY_QUESTIONS = [
    ("How is DTI assessed for a grocery business?", "POL-DTI"),
    ("Can we defer principal for a borrower 45 days past due?", "POL-AMEND"),
    ("Who approves a 3 million dollar facility?", "POL-SECURITY"),
]


def questions(memo_dir: Path, n: int, seed: int) -> list[tuple[str, str, set]]:
    """Return (kind, question, right answers) for a sample of the corpus."""
    memos = [parse_memo(p)[:2] for p in sorted(memo_dir.glob("*.txt"))]
    by_name: dict[str, set] = defaultdict(set)
    for loan_id, borrower in memos:
        by_name[borrower].add(loan_id)
    out = []
    for loan_id, borrower in random.Random(seed).sample(memos, n):
        out.append(
            (
                "borrower",
                f"What did we decide for {borrower.title()}?",
                by_name[borrower],
            )
        )
        out.append(("loan id", f"What was approved for loan {loan_id}?", {loan_id}))
    out += [("policy", q, {pid}) for q, pid in POLICY_QUESTIONS]
    return out


def found(hits: list, right: set) -> bool:
    """Report whether any hit is one of the right documents."""
    return any((lid or who) in right for lid, who, _, _ in hits)


def main() -> None:
    """Score each kind of question with k-NN alone and with hybrid search."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memos", type=Path, default=Path("data/generated/memos"))
    p.add_argument("--n", type=int, default=25, help="memos to sample")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()

    store, runtime = get_store(), get_runtime()
    qs = questions(a.memos, a.n, a.seed)
    print(
        f"{'question':<10}{'count':>7}{'k-NN':>8}{'hybrid':>8}   (found in top {a.k})"
    )
    for kind in ("borrower", "loan id", "policy"):
        group = [(q, right) for k, q, right in qs if k == kind]
        row = [
            sum(
                found(store.search(runtime, q, k=a.k, hybrid=h), right)
                for q, right in group
            )
            for h in (False, True)
        ]
        print(f"{kind:<10}{len(group):>7}{row[0]:>8}{row[1]:>8}")


if __name__ == "__main__":
    main()
