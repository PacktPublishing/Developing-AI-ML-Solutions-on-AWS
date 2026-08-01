"""Pipeline step 1 (Processing): synthesize the monotone credit dataset and split it.

Writes train.csv and test.csv to two processing outputs; downstream steps read them
by S3 property reference.
"""

import os

import numpy as np
import pandas as pd

TRAIN_OUT = "/opt/ml/processing/train"
TEST_OUT = "/opt/ml/processing/test"


def main() -> None:
    """Generate a monotone credit dataset (higher util/dpd, lower income -> riskier)."""
    rng = np.random.default_rng(0)
    n = 4000
    util = rng.uniform(0, 1, n)
    dpd = rng.integers(0, 6, n)
    income = rng.uniform(20_000, 120_000, n)
    z = 2.5 * util + 0.4 * dpd - 2.0e-5 * income - 1.0
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    df = pd.DataFrame({"util": util, "dpd": dpd, "income": income, "default": y})

    cut = int(0.8 * n)
    os.makedirs(TRAIN_OUT, exist_ok=True)
    os.makedirs(TEST_OUT, exist_ok=True)
    df.iloc[:cut].to_csv(f"{TRAIN_OUT}/train.csv", index=False)
    df.iloc[cut:].to_csv(f"{TEST_OUT}/test.csv", index=False)
    print(f"prepare: {cut} train / {n - cut} test rows")


if __name__ == "__main__":
    main()
