# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["syne-tune", "xgboost", "scikit-learn", "pandas", "mlflow"]
# ///
"""Local hyperparameter tuning for the challenger — Syne Tune, no instance, no quota.

The cloud script aws/jobs/amt.py runs the search as managed SageMaker Automatic
Model Tuning. This runs the SAME search on your laptop with Syne Tune — the AMT
team's own open-source tuner (Apache-2.0). Its LocalBackend evaluates each trial as
a local subprocess and a model-based Bayesian scheduler (TPE) proposes the next
configuration: genuine Bayesian search, no random fallback, no Docker, no AWS.

Same ranges, seed, and budget as the managed run. Each trial trains the monotone
XGBoost challenger (tuning/trial.py) and reports validation_auc back through Syne
Tune; trials log to the same MLflow experiment when MLFLOW_TRACKING_URI is set.

Env: MLFLOW_TRACKING_URI (optional — the MLflow App ARN, or a sqlite:// path),
     SEED (default 42), MAX_JOBS (default 6, matches aws/jobs/amt.py).
Run: make amt   (or: MLFLOW_TRACKING_URI=<app-arn> uv run tuning/amt.py)
"""

import os

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import LocalBackend
from syne_tune.config_space import loguniform, randint
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import TPE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "runs-local", "input", "data")
SEED = int(os.environ.get("SEED", "42"))
MAX_JOBS = int(os.environ.get("MAX_JOBS", "6"))  # matches aws/jobs/amt.py

# The same ranges the managed AMT run searches (aws/jobs/amt.py); data_dir is a
# constant, forwarded to each trial as --data_dir.
config_space = {
    "max_depth": randint(3, 8),
    "n_estimators": randint(200, 600),
    "learning_rate": loguniform(0.01, 0.1),
    "data_dir": DATA,
}

scheduler = TPE(
    config_space, metric="validation_auc", do_minimize=False, random_seed=SEED
)

tuner = Tuner(
    trial_backend=LocalBackend(entry_point=os.path.join(HERE, "trial.py")),
    scheduler=scheduler,
    stop_criterion=StoppingCriterion(max_num_trials_completed=MAX_JOBS),
    n_workers=1,  # sequential, so trials on a laptop stay legible
    tuner_name="ch02-challenger-hpo",
)
tuner.run()

df = load_experiment(tuner.name).results
best = df.loc[df.validation_auc.idxmax()]
print(
    f"\nbest: validation_auc {best.validation_auc:.6f}  "
    f"(max_depth {int(best.config_max_depth)}, "
    f"n_estimators {int(best.config_n_estimators)}, "
    f"learning_rate {best.config_learning_rate:.6f})"
)
