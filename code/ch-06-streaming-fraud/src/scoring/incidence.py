# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "scipy", "matplotlib", "catboost"]
# ///
"""Fraud incidence modeling, the Poisson way: fit the fraud arrival rate and project the count.

First a fit check: per-minute fraud counts against a fitted Poisson (fraud is rare, so most
minutes see none, the classic Poisson shape). Then a projection: a CatBoost-Poisson regressor
predicts the hourly fraud count from volume, amount, and hour. The rate is what a lender sizes
the fraud desk against.

Usage:
  uv run scoring/incidence.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import poisson

# -------------------------------------------------------------------------------
# Plotting palette
# -------------------------------------------------------------------------------
CHAPTER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OBSERVED = "#3A5DFC"
FITTED = "#E4408A"
ACTUAL = "#4AA7CD"
PREDICTED = "#2CA25F"
plt.rcParams.update({"font.family": "Arial", "font.size": 13})


def calculate_chi_square(actual, predicted):
    """Return the chi-square statistic between actual and predicted counts."""
    return np.sum((actual - predicted) ** 2 / predicted)


def main() -> None:
    """Check the per-minute Poisson fit, project the hourly count with CatBoost-Poisson, plot both."""
    df = pd.read_csv(f"{CHAPTER_DIR}/data/split/train.csv", parse_dates=["event_time"])
    df["Hour"] = df["event_time"].dt.floor("h")
    hourly = (
        df.groupby("Hour")
        .agg(
            FraudCount=("is_fraud", "sum"),
            TxCount=("is_fraud", "count"),
            AvgAmount=("amount_usd", "mean"),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------------
    # Fit check with CatBoost: fit a Poisson regressor on the per-minute counts, read its
    # average rate, and compare the observed arrivals to Poisson at that rate. Fraud is rare
    # (lambda < 1), the classic rare-event shape with P(0) largest.
    # -------------------------------------------------------------------------------
    per_min = (
        df.assign(m=df["event_time"].dt.floor("min"))
        .groupby("m")
        .agg(FraudCount=("is_fraud", "sum"), TxCount=("is_fraud", "count"))
        .reset_index()
    )
    per_min["HourOfDay"] = per_min["m"].dt.hour
    per_min["LogTxCount"] = np.log(per_min["TxCount"].clip(lower=1e-2))
    minute_feats = ["LogTxCount", "HourOfDay"]
    minute_model = CatBoostRegressor(
        loss_function="Poisson",
        iterations=300,
        learning_rate=0.05,
        depth=4,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )
    minute_model.fit(
        Pool(
            per_min[minute_feats],
            per_min["FraudCount"],
            cat_features=[minute_feats.index("HourOfDay")],
        )
    )
    # the model's average predicted rate is the CatBoost-Poisson lambda for the arrival check
    lam = minute_model.predict(
        Pool(per_min[minute_feats], cat_features=[minute_feats.index("HourOfDay")])
    ).mean()
    counts = per_min["FraudCount"]
    k = np.arange(0, int(counts.max()) + 1)
    observed = np.array([(counts == i).mean() for i in k])

    plt.figure(figsize=(8, 4.5), dpi=150)
    ax = plt.gca()
    ax.bar(k, observed, width=0.9, alpha=0.85, color=OBSERVED, label="observed")
    ax.plot(
        k,
        poisson.pmf(k, lam),
        color=FITTED,
        lw=2.5,
        marker="o",
        ms=6,
        label=f"CatBoost-Poisson (lambda = {lam:.2f})",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(k)
    ax.set_xlabel("fraud events per minute", fontsize=14)
    ax.set_ylabel("probability", fontsize=14)
    ax.set_title("Per-minute fraud arrivals: observed vs CatBoost-Poisson", fontsize=15)
    ax.legend(frameon=False, fontsize=13)
    plt.tight_layout()
    os.makedirs(f"{CHAPTER_DIR}/artifacts", exist_ok=True)
    plt.savefig(f"{CHAPTER_DIR}/artifacts/poisson_fit.png")
    print(
        f"per-minute lambda {lam:.3f}, variance {counts.var():.3f}, P(0) {poisson.pmf(0, lam):.3f}"
    )

    # -------------------------------------------------------------------------------
    # Projection: a CatBoost-Poisson regressor predicts the hourly fraud count
    # -------------------------------------------------------------------------------
    base = hourly.copy()
    base["HourOfDay"] = base["Hour"].dt.hour
    base["LogAmount"] = np.log(base["AvgAmount"].clip(lower=1e-2))
    base["LogTxCount"] = np.log(base["TxCount"].clip(lower=1e-2))

    features = ["LogTxCount", "LogAmount", "HourOfDay"]
    model = CatBoostRegressor(
        loss_function="Poisson",
        iterations=200,
        learning_rate=0.1,
        depth=4,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(
        Pool(
            base[features],
            base["FraudCount"],
            cat_features=[features.index("HourOfDay")],
        )
    )
    base["Predicted"] = model.predict(
        Pool(base[features], cat_features=[features.index("HourOfDay")])
    )
    chi = calculate_chi_square(base["FraudCount"], base["Predicted"])

    plt.figure(figsize=(12, 4.5), dpi=150)
    ax = plt.gca()
    ax.plot(
        base["Hour"], base["FraudCount"], label="actual fraud count", color=ACTUAL, lw=2
    )
    ax.plot(
        base["Hour"],
        base["Predicted"],
        label=rf"CatBoost-Poisson ($\chi^2$ = {chi:.1f})",
        ls="--",
        color=PREDICTED,
        lw=2,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("hour", fontsize=14)
    ax.set_ylabel("fraud count", fontsize=14)
    ax.set_title("Actual vs predicted hourly fraud counts", fontsize=15)
    ax.legend(frameon=False, fontsize=13)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{CHAPTER_DIR}/artifacts/incidence_projection.png")
    print(f"chi-square CatBoost-Poisson {chi:.1f}")
    print("saved artifacts/poisson_fit.png and incidence_projection.png")


if __name__ == "__main__":
    main()
