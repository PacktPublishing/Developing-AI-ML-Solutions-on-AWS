# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Synthesize the chapter's credit-application dataset and its business spec.

The running example is a mid-sized bank moving its scoring off-prem onto
SageMaker. A data scientist wants to test a built-in XGBoost model against the
bank's established Weight-of-Evidence (WOE) logistic-regression scorecard. Both
models must obey the same business rule: risk moves monotonically with certain
features (a higher bureau score may never raise predicted default risk, a higher
debt-to-income may never lower it, and so on). That rule is regulatory, not
cosmetic, so it is captured once here, in feature_spec.json, and read by both
trainers — the scorecard gets monotonicity for free from monotone WOE binning,
the challenger has it imposed through XGBoost monotone_constraints.

The target is generated from a monotone logit so the signal genuinely respects
those directions; that keeps the comparison honest rather than accidental.

Usage:
  uv run data/generate_applications.py --rows 20000 --seed 7
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------
# The business spec
# -------------------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# The business spec. monotone: sign of the allowed relationship between the
# feature and predicted probability of default (+1 up, -1 down, 0 unconstrained),
# in scikit-learn / XGBoost convention. Categoricals carry no direction.
NUMERIC = {
    "annual_income": -1,
    "debt_to_income": +1,
    "bureau_score": -1,
    "credit_utilization": +1,
    "employment_length_years": -1,
    "loan_amount": +1,
    "age": 0,
}
CATEGORICAL = ["home_ownership", "loan_purpose", "employment_status"]
TARGET = "default"

# Category levels and a per-level risk offset (log-odds) used only to generate a
# realistic target; the models never see these numbers.
CAT_LEVELS = {
    "home_ownership": {"OWN": -0.35, "MORTGAGE": -0.15, "RENT": 0.30},
    "loan_purpose": {
        "home_improvement": -0.20,
        "car": -0.05,
        "debt_consolidation": 0.15,
        "medical": 0.25,
        "other": 0.10,
    },
    "employment_status": {
        "employed": -0.20,
        "retired": -0.05,
        "self_employed": 0.15,
        "unemployed": 0.55,
    },
}


# -------------------------------------------------------------------------------
# Data synthesis
# -------------------------------------------------------------------------------
def _standardize(x: np.ndarray) -> np.ndarray:
    """Center and scale to unit variance so log-odds weights are comparable."""
    return (x - x.mean()) / (x.std() + 1e-9)


def synthesize(rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw correlated application features and a monotone-consistent default flag."""
    age = rng.integers(21, 75, rows)
    employment_length_years = np.clip(rng.gamma(2.0, 3.0, rows), 0, 45).round(1)
    annual_income = np.clip(rng.lognormal(10.8, 0.5, rows), 12_000, 400_000).round(-2)
    loan_amount = np.clip(rng.lognormal(9.4, 0.6, rows), 1_000, 75_000).round(-2)
    debt_to_income = np.clip(rng.beta(2.0, 5.0, rows) * 60, 0, 60).round(1)
    bureau_score = np.clip(rng.normal(680, 70, rows), 300, 850).round().astype(int)
    credit_utilization = np.clip(rng.beta(2.0, 3.0, rows) * 100, 0, 100).round(1)

    home_ownership = rng.choice(
        list(CAT_LEVELS["home_ownership"]), rows, p=[0.25, 0.45, 0.30]
    )
    loan_purpose = rng.choice(
        list(CAT_LEVELS["loan_purpose"]), rows, p=[0.15, 0.20, 0.35, 0.15, 0.15]
    )
    employment_status = rng.choice(
        list(CAT_LEVELS["employment_status"]), rows, p=[0.68, 0.10, 0.15, 0.07]
    )

    # Monotone log-odds: each weight's sign matches the business direction above
    # (income and score push risk down, dti/utilization/delinquencies push it up).
    logit = (
        -2.1
        + 0.9 * _standardize(debt_to_income)
        + 1.1 * _standardize(credit_utilization)
        + 0.5 * _standardize(np.log(loan_amount))
        - 1.2 * _standardize(bureau_score.astype(float))
        - 0.7 * _standardize(np.log(annual_income))
        - 0.4 * _standardize(employment_length_years)
    )
    for col, arr in (
        ("home_ownership", home_ownership),
        ("loan_purpose", loan_purpose),
        ("employment_status", employment_status),
    ):
        logit = logit + np.array([CAT_LEVELS[col][v] for v in arr])

    p = 1.0 / (1.0 + np.exp(-logit))
    default = rng.binomial(1, p)

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "debt_to_income": debt_to_income,
            "bureau_score": bureau_score,
            "credit_utilization": credit_utilization,
            "employment_length_years": employment_length_years,
            "loan_amount": loan_amount,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "employment_status": employment_status,
            TARGET: default,
        }
    )


# -------------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------------
def main() -> None:
    """Generate the dataset, a train/test split, and the shared feature spec."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = synthesize(args.rows, rng)

    # Deterministic split so training runs are reproducible across worlds.
    test = df.sample(frac=args.test_frac, random_state=args.seed)
    train = df.drop(test.index)
    split_dir = os.path.join(OUT_DIR, "split")
    os.makedirs(split_dir, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "applications.csv"), index=False)
    train.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(split_dir, "test.csv"), index=False)

    spec = {
        "target": TARGET,
        "numeric_features": list(NUMERIC),
        "categorical_features": CATEGORICAL,
        # per-feature monotone direction; categoricals default to 0 (none)
        "monotone_constraints": {**NUMERIC, **{c: 0 for c in CATEGORICAL}},
    }
    with open(os.path.join(split_dir, "feature_spec.json"), "w") as fh:
        json.dump(spec, fh, indent=2)

    rate = df[TARGET].mean()
    print(
        f"Wrote {len(df)} rows (train={len(train)}, test={len(test)}), "
        f"default rate {rate:.1%}, spec -> split/feature_spec.json"
    )


if __name__ == "__main__":
    main()
