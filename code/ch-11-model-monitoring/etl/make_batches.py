# /// script
# dependencies = ["pandas"]
# ///
"""Build the reference and current batches the monitors compare.

Reuses the credit bureau data from chapter 2: a reference batch (the distribution
the model was trained on) and a current batch with injected drift -- an economic
downturn where utilization and debt-to-income climb and bureau scores slip. The
monitors should catch the shift in the features, the score, and the model's own
feature contributions.

Usage:
  uv run etl/make_batches.py --source ../ch-02-mlops/data/split/train.csv --out data/generated
"""

import argparse
from pathlib import Path

import pandas as pd


def inject_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df shifted as a downturn would shift a credit population."""
    out = df.copy()
    out["credit_utilization"] = (out["credit_utilization"] * 1.35).clip(upper=100)
    out["debt_to_income"] = (out["debt_to_income"] * 1.25).clip(upper=100)
    out["bureau_score"] = (out["bureau_score"] - 45).clip(lower=300)
    out["annual_income"] = out["annual_income"] * 0.9
    # more self-employed applicants apply when times are hard
    mask = out.sample(frac=0.15, random_state=0).index
    out.loc[mask, "employment_status"] = "self_employed"
    return out


def main() -> None:
    """Split the source into reference and current (drifted) batches."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=Path("../ch-02-mlops/data/split/train.csv")
    )
    p.add_argument("--out", type=Path, default=Path("data/generated"))
    a = p.parse_args()

    df = pd.read_csv(a.source)
    ref = df.sample(frac=0.6, random_state=1)
    current = df.drop(ref.index)
    current = inject_drift(current)

    a.out.mkdir(parents=True, exist_ok=True)
    ref.to_csv(a.out / "reference.csv", index=False)
    current.to_csv(a.out / "current.csv", index=False)
    print(
        f"reference: {len(ref)} rows, current: {len(current)} rows (drifted) -> {a.out}"
    )


if __name__ == "__main__":
    main()
