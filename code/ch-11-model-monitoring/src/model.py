"""Train and load a CatBoost credit model and calculate default probabilities.

The model uses numeric and categorical credit bureau features. The monitoring
code also uses its split borders and native SHAP values.
"""

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


DEFAULTS = {"iterations": 300, "depth": 5, "learning_rate": 0.1}


def train(reference: pd.DataFrame, params: dict | None = None) -> CatBoostClassifier:
    """Fit the scorecard on the reference batch, honouring SageMaker hyperparameters."""
    p = {**DEFAULTS, **(params or {})}
    model = CatBoostClassifier(
        iterations=int(p["iterations"]),
        depth=int(p["depth"]),
        learning_rate=float(p["learning_rate"]),
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
