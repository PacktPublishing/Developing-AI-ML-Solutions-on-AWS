# /// script
# dependencies = ["catboost", "pandas"]
# ///
"""The scorecard: a CatBoost classifier over the credit bureau features.

CatBoost handles the categorical features natively and exposes feature importances
and SHAP values directly, which the drift monitors lean on. train() fits on the
reference batch; score() returns the probability of default the monitors watch.

Usage:
  uv run src/model.py --reference data/generated/reference.csv --out data/generated/scorecard.cbm
"""

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

NUMERIC = [
    "age",
    "annual_income",
    "debt_to_income",
    "bureau_score",
    "credit_utilization",
    "employment_length_years",
    "loan_amount",
]
CATEGORICAL = ["home_ownership", "loan_purpose", "employment_status"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "default"


def pool(df: pd.DataFrame) -> Pool:
    """Wrap a frame as a CatBoost Pool with the categorical features declared."""
    return Pool(df[FEATURES], label=df.get(TARGET), cat_features=CATEGORICAL)


def train(reference: pd.DataFrame) -> CatBoostClassifier:
    """Fit the scorecard on the reference batch."""
    model = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.1,
        loss_function="Logloss",
        verbose=False,
        random_seed=1,
    )
    model.fit(pool(reference))
    return model


def score(model: CatBoostClassifier, df: pd.DataFrame) -> pd.Series:
    """Return the probability of default for each row."""
    return pd.Series(
        model.predict_proba(df[FEATURES])[:, 1], index=df.index, name="score"
    )


def load(path: str | Path) -> CatBoostClassifier:
    """Load a saved scorecard."""
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def main() -> None:
    """Train the scorecard on the reference batch and save it."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", type=Path, default=Path("data/generated/reference.csv")
    )
    p.add_argument("--out", type=Path, default=Path("data/generated/scorecard.cbm"))
    a = p.parse_args()

    model = train(pd.read_csv(a.reference))
    model.save_model(str(a.out))
    top = sorted(zip(FEATURES, model.feature_importances_), key=lambda kv: -kv[1])[:5]
    print(
        "trained scorecard; top features:", [f"{name} {imp:.1f}" for name, imp in top]
    )
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
