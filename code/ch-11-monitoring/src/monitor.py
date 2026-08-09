# /// script
# dependencies = ["catboost", "pandas", "numpy"]
# ///
"""The drift monitors: Population Stability Index and SHAP attribution drift.

PSI compares a current distribution to the reference the model was trained on, per
feature and on the score itself: < 0.1 stable, 0.1-0.25 moderate, > 0.25 major.
The attribution monitor mirrors SageMaker Clarify's feature-attribution drift: it
ranks features by mean absolute SHAP contribution and reports the NDCG between the
reference ranking and the current one (1.0 = unchanged). run() returns the metrics
and the violations a monitoring schedule would raise.

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

PSI_MAJOR = 0.25
NDCG_MIN = 0.90


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    """Return the Population Stability Index between two binned distributions."""
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def numeric_psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """PSI for a numeric column, binned by the reference deciles."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    return _psi(ref_pct, cur_pct)


def categorical_psi(ref: pd.Series, cur: pd.Series) -> float:
    """PSI for a categorical column, over the union of categories."""
    cats = sorted(set(ref) | set(cur))
    ref_pct = np.array([(ref == c).mean() for c in cats])
    cur_pct = np.array([(cur == c).mean() for c in cats])
    return _psi(ref_pct, cur_pct)


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


def run(model, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run every monitor and return metrics plus the violations to raise."""
    metrics: dict = {"feature_psi": {}}
    violations = []

    for col in FEATURES:
        psi = (
            numeric_psi(reference[col], current[col])
            if col in NUMERIC
            else categorical_psi(reference[col], current[col])
        )
        metrics["feature_psi"][col] = round(psi, 3)
        if psi > PSI_MAJOR:
            violations.append(
                {
                    "monitor": "data-quality",
                    "feature": col,
                    "metric": "psi",
                    "value": round(psi, 3),
                    "threshold": PSI_MAJOR,
                }
            )

    metrics["score_psi"] = round(
        numeric_psi(score(model, reference), score(model, current)), 3
    )
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
    p.add_argument("--model", type=Path, default=Path("data/generated/scorecard.cbm"))
    a = p.parse_args()

    report = run(load(a.model), pd.read_csv(a.reference), pd.read_csv(a.current))
    print(json.dumps(report, indent=2))
    print(f"\n{len(report['violations'])} violation(s)")


if __name__ == "__main__":
    main()
