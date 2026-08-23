# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "catboost", "scikit-learn", "psycopg2-binary"]
# ///
"""Score-drift monitoring: seed the training reference, the live scores, and a drift attribution.

Writes three warehouse tables the Grafana board reads: score_reference (the fixed training
score distribution), score_monitor (live scores), and drift_attribution (a per-feature Shapley
attribution of the score drift). The score is a percentile-among-legit score raised to a power
(policy.FPRCalibrator; the default gamma puts 900 at a 2% false-positive rate, matching Amazon
Fraud Detector's published table), and the attribution is ScoreDriftAttributor, the label-free
virtual-drift side of Edakunni et al., "Explaining Drift using Shapley Values". The live stream is
the held-out test set spread over 24 hours plus a simulated fraud-campaign burst, so the drift is
visible; replace it with real live scores.

Usage:
  uv run scoring/monitoring.py
"""

import json
import os

import numpy as np
import pandas as pd
import psycopg2
from catboost import CatBoostClassifier
from drift import ScoreDriftAttributor
from policy import FPRCalibrator
from psycopg2.extras import execute_values

CHAPTER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
WAREHOUSE_JDBC = os.environ.get(
    "WAREHOUSE_JDBC", "jdbc:redshift://localhost:5439/fraud"
)
WAREHOUSE_USER = os.environ.get("WAREHOUSE_USER", "analyst")
WAREHOUSE_PASSWORD = os.environ.get("WAREHOUSE_PASSWORD", "analyst")
BUCKET = 100  # score-histogram bucket width on the 0-1000 axis
SEED = 6


def warehouse():
    """Open a connection to the local warehouse (or Redshift on AWS) from the JDBC env vars."""
    hostport, db = WAREHOUSE_JDBC.removeprefix("jdbc:redshift://").rsplit("/", 1)
    dsn = f"postgresql://{WAREHOUSE_USER}:{WAREHOUSE_PASSWORD}@{hostport}/{db}?sslmode=disable"
    return psycopg2.connect(dsn)


def main() -> None:
    """Score train and held-out test, build the reference/live/attribution tables, and seed them."""
    with open(f"{CHAPTER_DIR}/artifacts/model_meta.json") as f:
        features = json.load(f)["features"]
    model = CatBoostClassifier()
    model.load_model(f"{CHAPTER_DIR}/artifacts/model.cbm")
    train = pd.read_csv(
        f"{CHAPTER_DIR}/data/split/train.csv", parse_dates=["event_time"]
    )
    test = pd.read_csv(f"{CHAPTER_DIR}/data/split/test.csv", parse_dates=["event_time"])
    rng = np.random.default_rng(SEED)

    # the score is a training artifact: fit the calibrator's reference on the training scores
    calibrator = FPRCalibrator().fit(
        model.predict_proba(train[features])[:, 1], train["is_fraud"].values
    )

    # live = held-out test spread over 24h + a fraud-campaign burst in the last 90 minutes
    now = pd.Timestamp.utcnow().tz_localize(None)
    test = test.copy()
    test["event_time"] = [
        now - pd.Timedelta(hours=float(h)) for h in rng.uniform(1.5, 24.0, len(test))
    ]
    fraud_pool = pd.concat([train, test])[lambda d: d["is_fraud"] == 1]
    burst = fraud_pool.sample(1500, replace=True, random_state=SEED).copy()
    burst["event_time"] = [
        now - pd.Timedelta(minutes=int(x)) for x in rng.integers(0, 90, len(burst))
    ]
    live = pd.concat([test, burst], ignore_index=True)

    def score(df):
        return calibrator.transform(model.predict_proba(df[features])[:, 1]).ravel()

    ref = (
        pd.Series((score(train) // BUCKET) * BUCKET)
        .value_counts(normalize=True)
        .sort_index()
        .reindex(range(0, 1000, BUCKET), fill_value=0.0)
    )
    live_scores = score(live)
    attrib = (
        ScoreDriftAttributor(model, calibrator, features).fit(train).attribute(live)
    )
    print(
        f"total mean-score drift {attrib.sum():.1f}; top: "
        + ", ".join(f"{f} {v:+.1f}" for f, v in attrib.head(3).items())
    )

    with warehouse() as conn, conn.cursor() as cur:
        for t in ("score_reference", "score_monitor", "drift_attribution"):
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(
            "CREATE TABLE score_reference (bucket INT, train_share DOUBLE PRECISION)"
        )
        cur.execute("CREATE TABLE score_monitor (score INT, event_time TIMESTAMP)")
        cur.execute(
            "CREATE TABLE drift_attribution (feature VARCHAR(40), contribution DOUBLE PRECISION)"
        )
        execute_values(
            cur,
            "INSERT INTO score_reference VALUES %s",
            [(int(b), float(x)) for b, x in zip(ref.index, ref.values.round(6))],
        )
        execute_values(
            cur,
            "INSERT INTO score_monitor VALUES %s",
            [
                (int(s), t.to_pydatetime())
                for s, t in zip(live_scores, live["event_time"])
            ],
            page_size=1000,
        )
        execute_values(
            cur,
            "INSERT INTO drift_attribution VALUES %s",
            [(f, round(float(v), 3)) for f, v in attrib.items()],
        )
    print(
        f"seeded score_reference({len(ref)}), score_monitor({len(live_scores)}), drift_attribution({len(attrib)})"
    )


if __name__ == "__main__":
    main()
