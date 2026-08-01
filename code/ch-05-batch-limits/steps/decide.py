# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "pandas"]
# ///
"""Turn scores into limit decisions.

Usage:
  uv run steps/decide.py
"""

import io

import pandas as pd
from common import BUCKET, PREFIX, s3_client

INCREASE_PD = 0.05  # low risk
INCREASE_UTILIZATION = 0.60  # and the limit is working hard
INCREASE_STEP = 0.20  # +20%
DECREASE_PD = 0.25  # high risk
DECREASE_STEP = 0.30  # -30%


def decide(row: pd.Series) -> tuple[str, float]:
    """One customer's decision and new limit."""
    if row.pd_12m < INCREASE_PD and row.utilization > INCREASE_UTILIZATION:
        return "INCREASE", round(row.current_limit * (1 + INCREASE_STEP), -1)
    if row.pd_12m > DECREASE_PD:
        return "DECREASE", round(row.current_limit * (1 - DECREASE_STEP), -1)
    return "KEEP", row.current_limit


def main() -> None:
    """Apply the limit rules to the scored book and stage the decisions."""
    s3 = s3_client()
    scored = pd.read_csv(
        io.BytesIO(
            s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/scores.csv")["Body"].read()
        )
    )
    decisions = scored.apply(decide, axis=1, result_type="expand")
    scored["decision"], scored["new_limit"] = decisions[0], decisions[1]

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{PREFIX}/decisions.csv",
        Body=scored.to_csv(index=False).encode(),
    )
    counts = scored["decision"].value_counts().to_dict()
    print(f"decisions staged -> s3://{BUCKET}/{PREFIX}/decisions.csv {counts}")


if __name__ == "__main__":
    main()
