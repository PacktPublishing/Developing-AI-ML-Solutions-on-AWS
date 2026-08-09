# /// script
# dependencies = []
# ///
"""Score the classifier honestly: accuracy, macro-F1, and a multiclass Brier score.

We report the raw multiclass Brier score, not the binary "Brier Index" whose
100/50/0 anchors are calibrated to the two-class case and drift with the number of
classes. With hard (one-hot) predictions the multiclass Brier degenerates to twice
the error rate, so we also report a skill score against a uniform baseline, which
stays interpretable as the number of classes K changes. Soft probability outputs
would give a non-degenerate Brier; add a "probs" field to the predictions to use
them here.

Usage:
  uv run src/evaluate.py --dataset conversations
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _prf(preds: list[dict], labels: list[str]) -> float:
    """Return macro-averaged F1 over the label set."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
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


def _brier(preds: list[dict], k: int) -> tuple[float, float]:
    """Return (multiclass Brier from one-hot predictions, skill score vs. uniform)."""
    # one-hot prediction vs one-hot truth: 0 when correct, 2 when wrong -> 2 * error
    errors = sum(1 for r in preds if r["pred"] != r["label"])
    brier = 2 * errors / len(preds)
    brier_uniform = (k - 1) / k  # Brier of a uniform 1/K forecast over K classes
    skill = 1 - brier / brier_uniform if brier_uniform else 0.0
    return brier, skill


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
    k = len(labels)

    acc = sum(1 for r in preds if r["pred"] == r["label"]) / len(preds)
    macro_f1 = _prf(preds, labels)
    brier, skill = _brier(preds, k)

    print(f"dataset:    {a.dataset}  (n={len(preds)}, K={k} classes)")
    print(f"accuracy:   {acc:.3f}")
    print(f"macro-F1:   {macro_f1:.3f}")
    print(f"Brier:      {brier:.3f}   (multiclass, one-hot -> 2*error; range [0, 2])")
    print(
        f"skill:      {skill:.3f}   (1 - Brier/Brier_uniform, uniform = {(k - 1) / k:.3f})"
    )


if __name__ == "__main__":
    main()
