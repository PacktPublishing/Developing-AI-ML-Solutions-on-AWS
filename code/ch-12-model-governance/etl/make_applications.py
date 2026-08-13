# /// script
# dependencies = ["pandas"]
# ///
"""Build a batch of applications to route through the gateway.

Reuses chapter 2's held-out applications, drops the label, and gives each one a stable
loanId -- the key the rollout split buckets on -- so the same application always lands
with the same model.

Usage:
  uv run etl/make_applications.py --n 300 --out data/applications.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    """Write n labelled-free applications, each with a loanId, as JSONL."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=Path("../ch-02-mlops/data/split/test.csv")
    )
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("data/applications.jsonl"))
    a = p.parse_args()

    df = pd.read_csv(a.source).drop(columns=["default"], errors="ignore").head(a.n)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for i, row in enumerate(df.to_dict(orient="records")):
            fh.write(json.dumps({"loanId": f"L{i:06d}", **row}) + "\n")
    print(f"{len(df)} applications -> {a.out}")


if __name__ == "__main__":
    main()
