# /// script
# dependencies = []
# ///
"""Score the classifier: accuracy, macro-F1, the multiclass Brier, and the Brier Index.

The Brier Index (Forecasting Research Institute, 2026) rescales a binary Brier
score to 100 * (1 - sqrt(Brier)) so higher is better, with 100 = perfect, 50 =
always forecasting the base event, 0 = maximally wrong. Its anchors assume a 50%
base rate, so we apply it one-vs-rest -- one binary problem per class -- and also
report the **Adjusted Brier Index**, which references each class's own base rate p:

    Adjusted Brier Index = 100 - 50 * sqrt(Brier / (p * (1 - p)))

That is what makes it fair under class imbalance: a rare class with a low raw Brier
is not automatically "good", it is judged against p * (1 - p). The indices are only
meaningful with soft probabilities -- add a "probs" field to the predictions (see
classify.py --probs); with hard labels the Brier is just twice the error rate.

Usage:
  uv run src/evaluate.py --dataset conversations
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _prob_vector(r: dict, labels: list[str]) -> list[float]:
    """Return the predicted probability per class -- soft if present, else one-hot."""
    probs = r.get("probs")
    if probs:
        return [float(probs.get(label, 0.0)) for label in labels]
    return [1.0 if r["pred"] == label else 0.0 for label in labels]


def _macro_f1(preds: list[dict], labels: list[str]) -> float:
    """Return macro-averaged F1 over the label set."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in preds:
        if r["pred"] == r["label"]:
            tp[r["label"]] += 1
        else:
            fp[r["pred"]] += 1
            fn[r["label"]] += 1
    f1s = []
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / len(f1s)


def _brier_indices(preds: list[dict], labels: list[str]) -> tuple[list[dict], float]:
    """Return per-class one-vs-rest (base rate, Brier, Brier Index, Adjusted Index) and the multiclass Brier."""
    n = len(preds)
    rows = []
    multiclass = 0.0
    for j, label in enumerate(labels):
        base = sum(1 for r in preds if r["label"] == label) / n
        brier = (
            sum(
                (_prob_vector(r, labels)[j] - (1.0 if r["label"] == label else 0.0))
                ** 2
                for r in preds
            )
            / n
        )
        multiclass += brier
        bi = 100 * (1 - math.sqrt(brier))
        baseline = base * (1 - base)
        adj = 100 - 50 * math.sqrt(brier / baseline) if baseline > 0 else float("nan")
        rows.append(
            {"label": label, "base": base, "brier": brier, "bi": bi, "adj": adj}
        )
    return rows, multiclass


def main() -> None:
    """Print the classification metrics for a scored dataset."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="conversations")
    p.add_argument("--data", type=Path, default=Path("data/generated"))
    a = p.parse_args()

    preds = [
        json.loads(line)
        for line in (a.data / f"{a.dataset}.predictions.jsonl").read_text().splitlines()
    ]
    labels = json.loads((a.data / f"{a.dataset}.labels.json").read_text())
    soft = any(r.get("probs") for r in preds)

    acc = sum(1 for r in preds if r["pred"] == r["label"]) / len(preds)
    rows, multiclass = _brier_indices(preds, labels)
    valid = [r for r in rows if not math.isnan(r["adj"])]
    macro_bi = sum(r["bi"] for r in rows) / len(rows)
    macro_adj = sum(r["adj"] for r in valid) / len(valid) if valid else float("nan")

    print(
        f"dataset:        {a.dataset}  (n={len(preds)}, K={len(labels)} classes, {'soft' if soft else 'hard'} predictions)"
    )
    print(f"accuracy:       {acc:.3f}    macro-F1: {_macro_f1(preds, labels):.3f}")
    print(f"Brier score:    {multiclass:.3f}   (multiclass, lower is better)")
    print(
        f"Brier Index:    {macro_bi:.1f}%   (macro one-vs-rest, higher is better; 50%-base-rate anchor)"
    )
    print(
        f"Adjusted Index: {macro_adj:.1f}%   (macro, referenced to each class's own base rate)"
    )
    if not soft:
        print(
            "note: hard predictions -> Brier is 2*error; add --probs to classify for a calibrated Brier"
        )
    print("\nper class:   base    Brier   BrierIdx  AdjIdx")
    for r in rows:
        print(
            f"  {r['label']:20} {r['base']:.2f}   {r['brier']:.3f}   {r['bi']:5.1f}%   {r['adj']:5.1f}%"
        )


if __name__ == "__main__":
    main()
