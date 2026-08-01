"""Pipeline step 2 (Processing): train on labeled history, score the shortlist.

History comes from the warehouse; the shortlist arrives as a processing
input. Scores and the trained model go to two processing outputs.
"""

import os

import pandas as pd
import psycopg2
from catboost import CatBoostClassifier

IN = "/opt/ml/processing/input"
SCORES_OUT = "/opt/ml/processing/scores"
MODEL_OUT = "/opt/ml/processing/model"


def main() -> None:
    """Fit on the labeled past, score the eligible book."""
    features = os.environ["FEATURES"].split(",")
    with psycopg2.connect(os.environ["WAREHOUSE_DSN"]) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(features)}, default_12m FROM customers "
            "WHERE default_12m IS NOT NULL"
        )
        history = pd.DataFrame(cur.fetchall(), columns=[*features, "default_12m"])

    eligible = pd.read_csv(f"{IN}/eligible.csv")

    model = CatBoostClassifier(iterations=300, random_seed=5, verbose=0)
    model.fit(history[features], history["default_12m"].astype(int))

    scored = eligible[["customer_id", "current_limit", "utilization"]].copy()
    scored["pd_12m"] = model.predict_proba(eligible[features])[:, 1].round(6)

    os.makedirs(SCORES_OUT, exist_ok=True)
    os.makedirs(MODEL_OUT, exist_ok=True)
    scored.to_csv(f"{SCORES_OUT}/scores.csv", index=False)
    model.save_model(f"{MODEL_OUT}/model.cbm")
    print(f"scored {len(scored)} customers (mean pd {scored['pd_12m'].mean():.3f})")


if __name__ == "__main__":
    main()
