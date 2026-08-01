"""Pipeline step 3 (Processing): turn scores into limit decisions."""

import os

import pandas as pd

IN = "/opt/ml/processing/input"
OUT = "/opt/ml/processing/output"

INCREASE_PD = 0.05
INCREASE_UTILIZATION = 0.60
INCREASE_STEP = 0.20
DECREASE_PD = 0.25
DECREASE_STEP = 0.30


def decide(row: pd.Series) -> tuple[str, float]:
    """One customer's decision and new limit."""
    if row.pd_12m < INCREASE_PD and row.utilization > INCREASE_UTILIZATION:
        return "INCREASE", round(row.current_limit * (1 + INCREASE_STEP), -1)
    if row.pd_12m > DECREASE_PD:
        return "DECREASE", round(row.current_limit * (1 - DECREASE_STEP), -1)
    return "KEEP", row.current_limit


def main() -> None:
    """Apply the limit rules to the scored book."""
    scored = pd.read_csv(f"{IN}/scores.csv")
    decisions = scored.apply(decide, axis=1, result_type="expand")
    scored["decision"], scored["new_limit"] = decisions[0], decisions[1]

    os.makedirs(OUT, exist_ok=True)
    scored.to_csv(f"{OUT}/decisions.csv", index=False)
    print(scored["decision"].value_counts().to_dict())


if __name__ == "__main__":
    main()
