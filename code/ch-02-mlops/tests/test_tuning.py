"""Tests for the local Syne Tune search: the trial contract, parity, and a mini run.

Everything runs offline — trials are plain subprocesses, so no Docker and no
AWS credentials are involved. The mini search exercises the same LocalBackend +
TPE combination tuning/amt.py uses, just with a tiny dataset and two trials.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TUNING = Path(__file__).resolve().parent.parent / "tuning"
TRIAL = TUNING / "trial.py"
LOCAL_AMT = TUNING / "amt.py"
CLOUD_AMT = TUNING.parent / "aws" / "jobs" / "amt.py"


# -------------------------------------------------------------------------------
# Synthetic data
# -------------------------------------------------------------------------------
def make_dataset(root: Path, rows: int = 240, seed: int = 7) -> Path:
    """Lay out a tiny train/validation split shaped like the chapter's channels."""
    rng = np.random.default_rng(seed)
    score = rng.integers(300, 850, size=rows)
    utilization = rng.uniform(0, 1, size=rows)
    state = rng.choice(["CA", "NY", "TX"], size=rows)
    logit = 4.0 - 0.01 * score + 2.0 * utilization
    target = (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "bureau_score": score,
            "utilization": utilization,
            "state": state,
            "default_flag": target,
        }
    )
    spec = {
        "target": "default_flag",
        "numeric_features": ["bureau_score", "utilization"],
        "categorical_features": ["state"],
        "monotone_constraints": {"bureau_score": -1, "utilization": 1, "state": 0},
    }
    train_dir = root / "train"
    val_dir = root / "validation"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    split = int(rows * 0.75)
    df.iloc[:split].to_csv(train_dir / "train.csv", index=False)
    df.iloc[split:].to_csv(val_dir / "test.csv", index=False)
    (train_dir / "feature_spec.json").write_text(json.dumps(spec))
    return root


# -------------------------------------------------------------------------------
# The trial script
# -------------------------------------------------------------------------------
def test_trial_trains_and_reports_auc(tmp_path, monkeypatch):
    """One trial subprocess trains the challenger and reports validation_auc."""
    data = make_dataset(tmp_path / "data")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    result = subprocess.run(
        [
            sys.executable,
            str(TRIAL),
            "--max_depth",
            "3",
            "--n_estimators",
            "20",
            "--learning_rate",
            "0.1",
            "--data_dir",
            str(data),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    match = re.search(r"validation_auc[\"']?[:=] ?([0-9.]+)", result.stdout)
    assert match, f"no validation_auc in trial output:\n{result.stdout}"
    auc = float(match.group(1))
    assert 0.0 <= auc <= 1.0
    # The synthetic signal is monotone and learnable; the model must beat chance.
    assert auc > 0.5


# -------------------------------------------------------------------------------
# Local and cloud search the same space
# -------------------------------------------------------------------------------
def test_search_space_parity_with_cloud_amt():
    """tuning/amt.py and aws/jobs/amt.py declare the same ranges and objective."""
    local = LOCAL_AMT.read_text()
    cloud = CLOUD_AMT.read_text()
    for fragment in ("3, 8", "200, 600", "0.01, 0.1"):
        assert fragment in local, f"range ({fragment}) missing from local amt.py"
        assert fragment in cloud, f"range ({fragment}) missing from cloud amt.py"
    assert '"validation_auc"' in local or "'validation_auc'" in local
    assert "validation_auc" in cloud
    # The cloud run reads the objective through the CloudWatch regex contract.
    assert "validation_auc: ([0-9" in cloud


def test_trial_metric_matches_the_schedulers():
    """The trial reports the exact metric name both tuners optimize."""
    trial = TRIAL.read_text()
    assert "validation_auc=auc" in trial.replace(" ", "")


# -------------------------------------------------------------------------------
# A miniature search
# -------------------------------------------------------------------------------
def test_mini_tpe_search_completes(tmp_path, monkeypatch):
    """Two TPE trials run through LocalBackend and produce ranked results."""
    from syne_tune import StoppingCriterion, Tuner
    from syne_tune.backend import LocalBackend
    from syne_tune.config_space import loguniform, randint
    from syne_tune.experiments import load_experiment
    from syne_tune.optimizer.baselines import TPE

    data = make_dataset(tmp_path / "data")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    config_space = {
        "max_depth": randint(2, 4),
        "n_estimators": randint(10, 30),
        "learning_rate": loguniform(0.05, 0.2),
        "data_dir": str(data),
    }
    scheduler = TPE(
        config_space, metric="validation_auc", do_minimize=False, random_seed=7
    )
    tuner = Tuner(
        trial_backend=LocalBackend(entry_point=str(TRIAL)),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_num_trials_completed=2),
        n_workers=1,
        tuner_name="ch02-test-mini-search",
    )
    tuner.run()

    df = load_experiment(tuner.name).results
    done = df[df.validation_auc.notna()]
    assert len(done) >= 2
    best = done.loc[done.validation_auc.idxmax()]
    assert 0.0 <= float(best.validation_auc) <= 1.0
    assert 2 <= int(best.config_max_depth) <= 4
