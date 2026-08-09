# /// script
# dependencies = ["catboost", "pandas", "numpy"]
# ///
"""The drift monitors: Population Stability Index and SHAP attribution drift.

PSI compares a current distribution to the reference the model was trained on: < 0.1
stable, 0.1-0.25 moderate, > 0.25 major. Features are binned the ML way -- at the
scorecard's own CatBoost split borders (see psi.py), so PSI measures drift across the
boundaries the model actually uses -- while the score, which has no model borders, is
binned into reference quantiles. The attribution monitor mirrors SageMaker Clarify's
feature-attribution drift: it ranks features by mean absolute SHAP contribution and
reports the NDCG between the reference ranking and the current one (1.0 = unchanged).
run() returns the metrics and the violations a monitoring schedule would raise.

Usage:
  uv run src/monitor.py --reference data/generated/reference.csv --current data/generated/current.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from model import CATEGORICAL, FEATURES, NUMERIC, load, score
from psi import PSI_MAJOR, PSIDetector

NDCG_MIN = 0.90


def _attributions(model, df: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP contribution per feature (CatBoost native SHAP)."""
    shap = model.get_feature_importance(
        Pool(df[FEATURES], cat_features=CATEGORICAL), type="ShapValues"
    )
    return pd.Series(np.abs(shap[:, :-1]).mean(axis=0), index=FEATURES)


def attribution_ndcg(ref_attr: pd.Series, cur_attr: pd.Series) -> float:
    """NDCG of the current feature-attribution ranking against the reference ranking."""
    order = ref_attr.sort_values(ascending=False).index
    gains = cur_attr[order].to_numpy()
    discount = 1 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discount))
    ideal = float(np.sum(np.sort(gains)[::-1] * discount))
    return dcg / ideal if ideal else 1.0


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
    metrics: dict = {"feature_psi": {}}
    violations = []

    # Feature PSI the ML way: bin each feature at the model's own split borders.
    detector = PSIDetector.from_catboost(model, reference, NUMERIC, CATEGORICAL)
    metrics["feature_psi"] = detector.psi(current)
    for col, value in metrics["feature_psi"].items():
        if value > PSI_MAJOR:
            violations.append(
                {
                    "monitor": "data-quality",
                    "feature": col,
                    "metric": "psi",
                    "value": value,
                    "threshold": PSI_MAJOR,
                }
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
        _attributions(model, reference), _attributions(model, current)
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
