# /// script
# dependencies = ["catboost", "pandas", "numpy"]
# ///
"""The attribution-drift monitor: SHAP feature rankings compared with NDCG.

Mirrors SageMaker Clarify's feature-attribution drift: rank features by mean
absolute SHAP contribution and report the NDCG of the current ranking against
the reference ranking (1.0 = unchanged, below NDCG_MIN = drift). analysis_json
renders the ranking as Clarify's explainability report.

Usage:
  from attribution import attribution_ndcg, attributions
  ndcg = attribution_ndcg(attributions(model, reference), attributions(model, current))
"""

import numpy as np
import pandas as pd
from catboost import Pool
from model import CATEGORICAL, FEATURES

NDCG_MIN = 0.90


def attributions(model, df: pd.DataFrame) -> pd.Series:
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


def analysis_json(model, df: pd.DataFrame) -> dict:
    """Render Clarify's explainability report: global mean-abs SHAP per feature."""
    attr = attributions(model, df)
    return {
        "explanations": {
            "kernel_shap": {
                "default": {
                    "global_shap_values": {
                        f: round(float(attr[f]), 4) for f in FEATURES
                    },
                    "expected_value": round(
                        float(model.predict_proba(df[FEATURES])[:, 1].mean()), 4
                    ),
                }
            }
        }
    }
