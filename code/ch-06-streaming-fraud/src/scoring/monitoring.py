# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "catboost", "psycopg2-binary"]
# ///
"""Score-drift monitoring: seed the training reference, the live scores, and a DBShap attribution.

Writes three warehouse tables the Grafana board reads: score_reference (the fixed training
score distribution), score_monitor (live scores), and drift_attribution (a per-feature Shapley
attribution of the score drift, the label-free virtual-drift side of Edakunni et al., "Explaining
Drift using Shapley Values"). The live stream is the held-out test set spread over 24 hours plus
a simulated fraud-campaign burst, so the drift is visible; replace it with real live scores.

Usage:
  uv run scoring/monitoring.py
"""

import os
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
import psycopg2
from catboost import CatBoostClassifier
from policy import afd_score

CHAPTER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
WAREHOUSE_JDBC = os.environ.get(
    "WAREHOUSE_JDBC", "jdbc:redshift://localhost:5439/fraud"
)
WAREHOUSE_USER = os.environ.get("WAREHOUSE_USER", "analyst")
WAREHOUSE_PASSWORD = os.environ.get("WAREHOUSE_PASSWORD", "analyst")
BUCKET = 100  # score-histogram bucket width on the 0-1000 axis
COALITION_SAMPLES = 2000  # synthetic rows per coalition for the Shapley value function
SEED = 6


def warehouse():
    """Open a connection to the local warehouse (or Redshift on AWS) from the JDBC env vars."""
    hostport, db = WAREHOUSE_JDBC.removeprefix("jdbc:redshift://").rsplit("/", 1)
    dsn = f"postgresql://{WAREHOUSE_USER}:{WAREHOUSE_PASSWORD}@{hostport}/{db}?sslmode=disable"
    return psycopg2.connect(dsn)


def dbshap_attribution(model, features, train, live, legit, rng):
    """Shapley-attribute the mean-score drift to each feature's distribution shift.

    Value function is label-free: for a coalition, draw those features from the live marginal
    and the rest from the training marginal, score, and take the mean-score shift. The Shapley
    values are signed and sum to the total mean-score drift.
    """
    n = len(features)
    train_mat = np.column_stack(
        [rng.choice(train[f].values, COALITION_SAMPLES) for f in features]
    )
    live_mat = np.column_stack(
        [rng.choice(live[f].values, COALITION_SAMPLES) for f in features]
    )
    big = np.empty((2**n * COALITION_SAMPLES, n))
    for c in range(2**n):
        bits = np.array([(c >> j) & 1 for j in range(n)], dtype=bool)
        big[c * COALITION_SAMPLES : (c + 1) * COALITION_SAMPLES] = np.where(
            bits[None, :], live_mat, train_mat
        )
    scored = afd_score(
        model.predict_proba(pd.DataFrame(big, columns=features))[:, 1], legit
    )
    v = scored.reshape(2**n, COALITION_SAMPLES).mean(axis=1)
    v -= v[0]  # v(empty) = 0
    weight = {k: factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)}
    phi = np.zeros(n)
    for j in range(n):
        others = [i for i in range(n) if i != j]
        for k in range(len(others) + 1):
            for subset in combinations(others, k):
                s = sum(1 << i for i in subset)
                phi[j] += weight[k] * (v[s | (1 << j)] - v[s])
    return sorted(zip(features, phi), key=lambda t: -abs(t[1]))


def main() -> None:
    """Score train and held-out test, build the reference/live/attribution tables, and seed them."""
    import json

    with open(f"{CHAPTER_DIR}/artifacts/model_meta.json") as f:
        features = json.load(f)["features"]
    model = CatBoostClassifier()
    model.load_model(f"{CHAPTER_DIR}/artifacts/model.cbm")
    train = pd.read_csv(
        f"{CHAPTER_DIR}/data/split/train.csv", parse_dates=["event_time"]
    )
    test = pd.read_csv(f"{CHAPTER_DIR}/data/split/test.csv", parse_dates=["event_time"])
    rng = np.random.default_rng(SEED)

    # the legitimate-probability reference is fixed at training time; every score is read against it
    legit = model.predict_proba(train[features])[:, 1][train["is_fraud"].values == 0]

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

    ref = (
        pd.Series(
            (afd_score(model.predict_proba(train[features])[:, 1], legit) // BUCKET)
            * BUCKET
        )
        .value_counts(normalize=True)
        .sort_index()
        .reindex(range(0, 1000, BUCKET), fill_value=0.0)
    )
    live_scores = afd_score(model.predict_proba(live[features])[:, 1], legit)
    attrib = dbshap_attribution(model, features, train, live, legit, rng)
    print(
        f"total mean-score drift {sum(p for _, p in attrib):.1f}; top: "
        + ", ".join(f"{f} {p:+.1f}" for f, p in attrib[:3])
    )

    from psycopg2.extras import execute_values

    with warehouse() as conn, conn.cursor() as cur:
        for t in ("score_reference", "score_monitor", "drift_attribution"):
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute(
            "CREATE TABLE score_reference (bucket INT, train_share DOUBLE PRECISION)"
        )
        cur.execute("CREATE TABLE score_monitor (afd_score INT, event_time TIMESTAMP)")
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
            [(f, round(float(p), 3)) for f, p in attrib],
        )
    print(
        f"seeded score_reference({len(ref)}), score_monitor({len(live_scores)}), drift_attribution({len(attrib)})"
    )


if __name__ == "__main__":
    main()
