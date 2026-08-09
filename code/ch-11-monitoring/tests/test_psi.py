"""PSIDetector: quantile bins, model-border bins, and serialisation."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from psi import PSIDetector


def _frame(n: int, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Return a small synthetic batch; shift moves the x distribution to inject drift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(shift, 1, n),
            "y": rng.uniform(0, 100, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )


def test_psi_is_near_zero_against_itself():
    """The reference scored against itself is stable on every feature."""
    ref = _frame(4000, seed=1)
    det = PSIDetector.from_reference(ref, ["x", "y"], ["cat"], bins=10)
    assert all(v < 0.01 for v in det.psi(ref).values())


def test_psi_flags_the_shifted_feature_only():
    """A shift in x reads as major PSI on x while y stays stable."""
    det = PSIDetector.from_reference(_frame(4000, seed=1), ["x", "y"], ["cat"])
    psi = det.psi(_frame(4000, shift=1.5, seed=2))
    assert psi["x"] > 0.25
    assert psi["y"] < 0.1


def test_json_round_trip_preserves_scores(tmp_path):
    """A saved-and-loaded detector scores a batch identically."""
    ref, cur = _frame(2000, seed=1), _frame(2000, shift=1.0, seed=2)
    det = PSIDetector.from_reference(ref, ["x", "y"], ["cat"])
    path = tmp_path / "psi.json"
    det.save(path)
    assert PSIDetector.load(path).psi(cur) == det.psi(cur)


def test_catboost_bins_are_the_models_own_borders():
    """from_catboost bins each numeric feature at the model's split thresholds."""
    ref = _frame(4000, seed=1)
    y = ((ref["x"] + ref["y"] / 50) > 1).astype(int)
    model = CatBoostClassifier(iterations=40, depth=3, verbose=False, random_seed=0)
    model.fit(ref[["x", "y", "cat"]], y, cat_features=["cat"])

    det = PSIDetector.from_catboost(model, ref, ["x", "y"], ["cat"])
    idx = list(model.feature_names_).index("x")
    assert det.borders["x"] == [float(b) for b in model.get_borders()[idx]]
    assert det.binning == "catboost"
    # drift across those borders raises PSI above the stable reference
    assert det.psi(_frame(4000, shift=1.5, seed=2))["x"] > det.psi(ref)["x"]
