# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Synthesize the chapter's card-transaction stream and its training data.

The running example is a fintech whose card processor asks for a decision while
the terminal still shows "Processing Payment": approve or decline, within a
latency budget the cardholder never notices. The classifier behind that decision
is trained here on three families of features, the same families Revolut
describes for its card-fraud system: raw transaction attributes (amount, channel,
hour), user-versus-history comparisons (how far this amount sits from the
user's average, how fast it follows the previous one), and merchant-focused
aggregates (how much fraud this merchant has seen, how long it has existed).

Fraud is generated from a real logit over those features, so the signal the
model finds is genuine rather than accidental, and the label is rare the way
fraud is rare: a fraction of a percent of rows. The split is chronological —
train on the past, test on the future — because that is how a fraud model
meets production, and a random split would leak tomorrow's patterns into
yesterday's training set.

Outputs, under data/:
  stream/transactions.csv  the replay file for the producer (no label — live
                           events do not arrive labeled)
  labels.csv               ground truth per transaction_id, kept aside
  split/train.csv          labeled past, for the trainer
  split/test.csv           labeled future, at the production fraud ratio
  feature_spec.json        the model's input columns, read by the trainer

Usage:
  uv run data/generate_transactions.py --rows 50000 --seed 6
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

N_USERS = 2000
N_MERCHANTS = 300
CHANNELS = ["pos", "online", "atm"]
CATEGORIES = ["grocery", "electronics", "travel", "fashion", "gaming", "services"]

# The model's inputs, in the order every stage of the chapter uses them: the
# trainer fits on these columns, the consumer assembles them from the stream
# record, and the two must agree or the endpoint scores garbage.
FEATURES = [
    "amount_usd",
    "amount_over_user_avg",
    "seconds_since_last_txn",
    "is_first_merchant",
    "user_txn_count_24h",
    "merchant_fraud_rate",
    "merchant_age_days",
    "is_online",
    "is_night",
]


def generate(rows: int, seed: int) -> pd.DataFrame:
    """Draw users, merchants, and a chronological stream of transactions."""
    rng = np.random.default_rng(seed)

    # Users differ in how much they typically spend; merchants differ in how
    # much fraud they attract. A handful of risky merchants carry most of it,
    # which is what merchant-focused features exist to pick up.
    user_avg = rng.lognormal(mean=3.2, sigma=0.6, size=N_USERS)
    merchant_risk = rng.beta(0.08, 8.0, size=N_MERCHANTS)
    merchant_age = rng.integers(5, 3000, size=N_MERCHANTS)
    merchant_cat = rng.integers(0, len(CATEGORIES), size=N_MERCHANTS)

    user = rng.integers(0, N_USERS, size=rows)
    merchant = rng.integers(0, N_MERCHANTS, size=rows)
    amount = user_avg[user] * rng.lognormal(mean=0.0, sigma=0.7, size=rows)
    hour = rng.integers(0, 24, size=rows)
    channel = rng.choice(CHANNELS, size=rows, p=[0.55, 0.4, 0.05])
    seconds_since_last = rng.exponential(scale=6 * 3600, size=rows).clip(1, 7 * 86400)
    is_first_merchant = (rng.random(size=rows) < 0.15).astype(int)
    txn_count_24h = rng.poisson(lam=2.0, size=rows).clip(0, 30)

    df = pd.DataFrame(
        {
            "transaction_id": [f"tx-{i:07d}" for i in range(rows)],
            "user_id": [f"u-{u:05d}" for u in user],
            "merchant_id": [f"m-{m:04d}" for m in merchant],
            "merchant_category": [CATEGORIES[c] for c in merchant_cat[merchant]],
            "amount_usd": amount.round(2),
            "hour": hour,
            "channel": channel,
            "seconds_since_last_txn": seconds_since_last.round(0),
            "is_first_merchant": is_first_merchant,
            "user_txn_count_24h": txn_count_24h,
            "merchant_fraud_rate": merchant_risk[merchant].round(4),
            "merchant_age_days": merchant_age[merchant],
        }
    )
    df["amount_over_user_avg"] = (amount / user_avg[user]).round(3)
    df["is_online"] = (df["channel"] == "online").astype(int)
    df["is_night"] = df["hour"].isin(range(6)).astype(int)

    # The label: a logit over the same feature families the model will see.
    # Every term is a fraud pattern the chapter narrates — an amount far off
    # the user's own average, a merchant never used before, a rapid-fire
    # sequence, a risky merchant, an online purchase at night.
    z = (
        -8.5
        + 2.8 * np.log(df["amount_over_user_avg"].clip(lower=0.05))
        + 2.2 * df["is_first_merchant"]
        + 1.4 * df["is_online"]
        + 1.0 * df["is_night"]
        + 40.0 * df["merchant_fraud_rate"]
        - 0.30 * np.log1p(df["seconds_since_last_txn"])
    )
    p = 1.0 / (1.0 + np.exp(-z))
    df["is_fraud"] = (rng.random(size=rows) < p).astype(int)

    # A chronological clock: one event every few seconds, so the stream file
    # replays in time order and the split has a real past and future.
    step = rng.exponential(scale=3.0, size=rows).clip(0.2, 60)
    df["event_time"] = pd.Timestamp("2026-07-01T00:00:00Z") + pd.to_timedelta(
        np.cumsum(step), unit="s"
    )
    return df


def main() -> None:
    """Generate the stream file, the labels, and the chronological split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    rate = df["is_fraud"].mean()

    os.makedirs(f"{OUT_DIR}/stream", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/split", exist_ok=True)

    # The stream never carries the label; ground truth goes to its own file.
    df.drop(columns=["is_fraud"]).to_csv(
        f"{OUT_DIR}/stream/transactions.csv", index=False
    )
    df[["transaction_id", "is_fraud"]].to_csv(f"{OUT_DIR}/labels.csv", index=False)

    # Chronological 80/20: the model trains on the past and is judged on the
    # future, at the fraud ratio the future actually has.
    cut = int(len(df) * 0.8)
    df.iloc[:cut].to_csv(f"{OUT_DIR}/split/train.csv", index=False)
    df.iloc[cut:].to_csv(f"{OUT_DIR}/split/test.csv", index=False)

    # The contract with the trainer: which columns are model inputs.
    with open(f"{OUT_DIR}/feature_spec.json", "w") as f:
        json.dump({"features": FEATURES}, f, indent=2)

    print(
        f"{len(df)} transactions, fraud rate {rate:.4%} "
        f"({df['is_fraud'].sum()} fraudulent), split {cut}/{len(df) - cut}"
    )


if __name__ == "__main__":
    main()
