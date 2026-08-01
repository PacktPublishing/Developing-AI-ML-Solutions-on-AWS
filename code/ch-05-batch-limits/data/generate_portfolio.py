# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Synthesize the loan portfolio the batch run manages.

The running example is a lender's card book: thousands of customers with a
credit limit, a utilization, and a repayment history. Once a month a batch
run re-scores everyone, shortlists who is eligible for a limit change, and
applies the decisions — no customer asked for anything; the portfolio is
managed, not just observed.

The label is twelve-month default, generated from a real logit over the
behavior drivers (utilization, delinquency, tenure, affordability), so the
model the pipeline trains finds genuine signal. Older customers carry the
label (the past the model trains on); the newest vintage is unlabeled (the
book the run scores).

Outputs, under data/:
  portfolio.csv        every customer, labeled where history allows
  feature_spec.json    the model's input columns, read by the scoring step

Usage:
  uv run data/generate_portfolio.py --rows 20000 --seed 5
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    "current_limit",
    "utilization",
    "months_on_book",
    "dpd_30_count_12m",
    "monthly_income",
    "payment_ratio",
    "cash_advance_share",
]


def generate(rows: int, seed: int) -> pd.DataFrame:
    """Draw the portfolio: limits, behavior, and the default label."""
    rng = np.random.default_rng(seed)

    income = rng.lognormal(mean=8.1, sigma=0.5, size=rows)
    limit = (income * rng.uniform(0.5, 3.0, size=rows)).round(-2).clip(200)
    utilization = rng.beta(1.6, 2.4, size=rows)
    months = rng.integers(1, 96, size=rows)
    dpd = rng.poisson(lam=0.4, size=rows).clip(0, 12)
    payment_ratio = rng.beta(5, 2, size=rows)  # share of statement paid
    cash_share = rng.beta(1.2, 8, size=rows)  # cash advances, a risk tell

    df = pd.DataFrame(
        {
            "customer_id": [f"c-{i:06d}" for i in range(rows)],
            "current_limit": limit,
            "utilization": utilization.round(3),
            "months_on_book": months,
            "dpd_30_count_12m": dpd,
            "monthly_income": income.round(0),
            "payment_ratio": payment_ratio.round(3),
            "cash_advance_share": cash_share.round(3),
        }
    )

    # The label: a logit over the same behavior the model will see.
    z = (
        -4.2
        + 2.0 * df["utilization"]
        + 0.55 * df["dpd_30_count_12m"]
        - 2.2 * df["payment_ratio"]
        + 2.5 * df["cash_advance_share"]
        - 0.004 * df["months_on_book"]
    )
    p = 1.0 / (1.0 + np.exp(-z))
    df["default_12m"] = (rng.random(size=rows) < p).astype(int)

    # The newest vintage has no twelve-month outcome yet: that is the book
    # the batch run scores. Everyone older is training history.
    df.loc[df["months_on_book"] < 12, "default_12m"] = None
    return df


def main() -> None:
    """Generate the portfolio file and the feature spec."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    labeled = df["default_12m"].notna()
    df.to_csv(f"{OUT_DIR}/portfolio.csv", index=False)
    with open(f"{OUT_DIR}/feature_spec.json", "w") as f:
        json.dump({"features": FEATURES}, f, indent=2)
    print(
        f"{len(df)} customers, {labeled.sum()} labeled "
        f"(default rate {df.loc[labeled, 'default_12m'].mean():.2%}), "
        f"{(~labeled).sum()} to score"
    )


if __name__ == "__main__":
    main()
