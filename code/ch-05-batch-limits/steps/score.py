# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "pandas", "catboost"]
# ///
"""Train the behavior model on labeled history and score the shortlist.

Usage:
  uv run steps/score.py
"""

import io
import json

import pandas as pd
from catboost import CatBoostClassifier
from common import BUCKET, PREFIX, s3_client


def main() -> None:
    """Fit on labeled history, score the eligible book, stage both to S3."""
    with open("data/feature_spec.json") as f:
        features = json.load(f)["features"]

    history = pd.read_csv("data/portfolio.csv").dropna(subset=["default_12m"])
    s3 = s3_client()
    eligible = pd.read_csv(
        io.BytesIO(
            s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/eligible.csv")["Body"].read()
        )
    )

    model = CatBoostClassifier(iterations=300, random_seed=5, verbose=0)
    model.fit(history[features], history["default_12m"].astype(int))

    scored = eligible[["customer_id", "current_limit", "utilization"]].copy()
    scored["pd_12m"] = model.predict_proba(eligible[features])[:, 1].round(6)

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{PREFIX}/scores.csv",
        Body=scored.to_csv(index=False).encode(),
    )
    model.save_model("/tmp/ch05-model.cbm")
    with open("/tmp/ch05-model.cbm", "rb") as f:
        s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/model.cbm", Body=f.read())
    print(
        f"scored {len(scored)} customers -> s3://{BUCKET}/{PREFIX}/scores.csv "
        f"(mean pd {scored['pd_12m'].mean():.3f})"
    )


if __name__ == "__main__":
    main()
