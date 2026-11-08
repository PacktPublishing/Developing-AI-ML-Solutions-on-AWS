# /// script
# dependencies = ["catboost", "pandas", "numpy"]
# ///
"""Calculate feature PSI, score PSI and SHAP attribution drift.

Numeric feature bins use CatBoost split borders, with reference quantiles as a
fallback. Score PSI uses reference quantiles. Attribution drift uses the NDCG
function in attribution.py. Return metrics and threshold violations.

Usage:
  uv run src/monitor.py --reference data/generated/reference.csv --current data/generated/current.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from attribution import NDCG_MIN, attribution_ndcg, attributions
from model import CATEGORICAL, NUMERIC, load, score
from psi import PSI_MAJOR, PSIDetector


def run(
    model,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    current_scores: pd.Series | None = None,
) -> dict:
    """Run every monitor and return metrics plus the violations to raise.

    current_scores lets the caller pass the live scores from a Batch Transform job;
    when omitted the monitor scores the current batch itself.
    """
    if current_scores is None:
        current_scores = score(model, current)
    violations = []

    # Use model split borders for numeric feature PSI, with quantile fallback.
    detector = PSIDetector.from_catboost(model, reference, NUMERIC, CATEGORICAL)
    metrics: dict = {"feature_psi": detector.psi(current)}
    violations.extend(
        {
            "monitor": "data-quality",
            "feature": col,
            "metric": "psi",
            "value": value,
            "threshold": PSI_MAJOR,
        }
        for col, value in metrics["feature_psi"].items()
        if value > PSI_MAJOR
    )
    # Score PSI: the score has no model borders, so bin it into reference quantiles.
    score_ref = pd.DataFrame({"score": score(model, reference)})
    metrics["score_psi"] = PSIDetector.from_reference(score_ref, ["score"]).psi(
        pd.DataFrame({"score": current_scores})
    )["score"]
    if metrics["score_psi"] > PSI_MAJOR:
        violations.append(
            {
                "monitor": "model-quality",
                "feature": "score",
                "metric": "psi",
                "value": metrics["score_psi"],
                "threshold": PSI_MAJOR,
            }
        )

    ndcg = attribution_ndcg(
        attributions(model, reference), attributions(model, current)
    )
    metrics["attribution_ndcg"] = round(ndcg, 3)
    if ndcg < NDCG_MIN:
        violations.append(
            {
                "monitor": "clarify",
                "feature": "attribution",
                "metric": "ndcg",
                "value": round(ndcg, 3),
                "threshold": NDCG_MIN,
            }
        )

    return {"metrics": metrics, "violations": violations}


def constraint_violations(report: dict) -> dict:
    """Format threshold breaches for constraint_violations.json."""
    return {
        "violations": [
            {
                "feature_name": v["feature"],
                "constraint_check_type": v["monitor"],
                "description": f"{v['metric']} {v['value']} breached threshold {v['threshold']}",
            }
            for v in report["violations"]
        ]
    }


def main() -> None:
    """Run the monitors on the reference and current batches and print the report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", type=Path, default=Path("data/generated/reference.csv")
    )
    p.add_argument("--current", type=Path, default=Path("data/generated/current.csv"))
    p.add_argument("--model", type=Path, default=Path("runs-local/model/scorecard.cbm"))
    a = p.parse_args()

    report = run(load(a.model), pd.read_csv(a.reference), pd.read_csv(a.current))
    print(json.dumps(report, indent=2))
    print(f"\n{len(report['violations'])} violation(s)")


if __name__ == "__main__":
    main()
