# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "statsmodels", "scipy", "matplotlib", "catboost"]
# ///
"""Fraud incidence modeling, the Poisson way: fit the hourly fraud count and project it.

Two views, following the fraud-Poisson study on the chapter's own transactions. First a fit
check: the empirical hourly fraud count against a Poisson and a Negative Binomial. Then a
projection: a Negative Binomial and a CatBoost-Poisson predict the count from volume,
amount, and hour, compared by chi-square. The rate is what a lender sizes the fraud
desk against.

Usage:
  uv run scoring/incidence.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from catboost import CatBoostRegressor, Pool
from scipy.stats import nbinom, poisson

CHAPTER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
COLORS = ["#3A5DFC", "#4AA7CD", "#FF8EFF"]


def calculate_chi_square(actual, predicted):
    """Calculate chi-square statistic."""
    return np.sum((actual - predicted) ** 2 / predicted)


def main() -> None:
    """Aggregate hourly, check the Poisson/NB fit, and project the count with three models."""
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

    # ---------- Fit check: is fraud arrival Poisson? Test at a fine bucket where fraud is rare
    # (per minute, lambda < 1), the classic rare-event shape with P(0) largest. ----------
    per_min = (
        df.assign(m=df["event_time"].dt.floor("min")).groupby("m")["is_fraud"].sum()
    )
    lam = per_min.mean()
    var_min = per_min.var()
    alpha = max((var_min - lam) / (lam**2), 1e-6)  # NB dispersion by moments
    r = 1 / alpha
    p = r / (r + lam)

    k_values = np.arange(0, int(per_min.max()) + 1)
    observed = np.array([(per_min == k).mean() for k in k_values])
    poisson_prob = poisson.pmf(k_values, mu=lam)
    nb_prob = nbinom.pmf(k_values, r, p)

    plt.figure(figsize=(7, 4), dpi=150)
    ax = plt.gca()
    plt.bar(k_values, observed, width=0.9, alpha=0.8, label="Observed", color=COLORS[0])
    plt.plot(
        k_values,
        poisson_prob,
        color=COLORS[1],
        label="Poisson (inferred)",
        lw=2,
        marker="o",
        ms=4,
    )
    plt.plot(
        k_values,
        nb_prob,
        color=COLORS[2],
        label="Negative Binomial (inferred)",
        lw=2,
        marker="o",
        ms=4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(k_values)
    plt.xlabel("Fraud Events per Minute")
    plt.ylabel("Probability")
    plt.title("Per-Minute Fraud Arrivals: Observed vs Inferred Probability")
    plt.legend(framealpha=1, frameon=False)
    plt.tight_layout()
    os.makedirs(f"{CHAPTER_DIR}/artifacts", exist_ok=True)
    plt.savefig(f"{CHAPTER_DIR}/artifacts/poisson_fit.png")
    print(
        f"per-minute lambda {lam:.3f}, variance {var_min:.3f}, P(0) {poisson.pmf(0, lam):.3f}, NB alpha {alpha:.3f}"
    )

    # ---------- Projection: Negative Binomial and CatBoost-Poisson, compared by chi-square ----------
    base = hourly.copy()
    base["HourOfDay"] = base["Hour"].dt.hour
    base["AvgAmount"] = base["AvgAmount"].clip(lower=1e-2)
    base["TxCount"] = base["TxCount"].clip(lower=1e-2)
    base["LogAmount"] = np.log(base["AvgAmount"])
    base["LogTxCount"] = np.log(base["TxCount"])

    nb_model = smf.glm(
        "FraudCount ~ LogTxCount + LogAmount + C(HourOfDay)",
        data=base,
        family=sm.families.NegativeBinomial(),
    ).fit()
    base["NB_Pred"] = nb_model.predict(base)
    chi_nb = calculate_chi_square(base["FraudCount"], base["NB_Pred"])

    feat = ["LogTxCount", "LogAmount", "HourOfDay"]
    cat_features = [feat.index("HourOfDay")]
    cat_model = CatBoostRegressor(
        loss_function="Poisson",
        iterations=200,
        learning_rate=0.1,
        depth=4,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )
    cat_model.fit(Pool(base[feat], base["FraudCount"], cat_features=cat_features))
    base["CAT_Pred"] = cat_model.predict(Pool(base[feat], cat_features=cat_features))
    chi_cat = calculate_chi_square(base["FraudCount"], base["CAT_Pred"])

    plt.figure(figsize=(12, 4), dpi=150)
    plt.plot(
        base["Hour"],
        base["FraudCount"],
        label="Actual Fraud Count",
        color="#4AA7CD",
        lw=2,
    )
    plt.plot(
        base["Hour"],
        base["NB_Pred"],
        label=rf"Negative Binomial ($\chi^2$={chi_nb:.1f})",
        ls="--",
        color="#1f77b4",
        lw=2,
    )
    plt.plot(
        base["Hour"],
        base["CAT_Pred"],
        label=rf"CatBoost Poisson ($\chi^2$={chi_cat:.1f})",
        ls="-.",
        color="#2ca02c",
        lw=2,
    )
    plt.title("Actual vs Predicted Hourly Fraud Counts")
    plt.xlabel("Hour")
    plt.ylabel("Fraud Count")
    plt.legend(loc="upper right", fontsize=8)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{CHAPTER_DIR}/artifacts/incidence_projection.png")
    print(f"chi-square  NB {chi_nb:.1f}  CatBoost {chi_cat:.1f}")
    print("saved artifacts/poisson_fit.png and incidence_projection.png")


if __name__ == "__main__":
    main()
