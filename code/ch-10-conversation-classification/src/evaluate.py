"""Calculate accuracy, macro-F1 and Brier metrics for classification results.

Accuracy and macro-F1 include all records. Brier metrics include only records
with a valid forecast; coverage reports the proportion included.

For each class, the Brier Index is 100 * (1 - sqrt(Brier)). The Adjusted Brier
Index is 100 - 50 * sqrt(Brier / (p * (1 - p))), where p is the class frequency
among valid forecasts. The adjusted index is undefined when p is zero or one.
Multiclass Brier equals twice the error rate for valid one-hot predictions.

Usage:
  uv run src/evaluate.py --dataset conversations
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Per-class table columns: (heading, ClassScore field, format).
CLASS_COLUMNS = (
    ("base", "base", "{:.2f}"),
    ("Brier", "brier", "{:.3f}"),
    ("BrierIdx", "index", "{:.1f}%"),
    ("AdjIdx", "adjusted", "{:.1f}%"),
)


@dataclass(frozen=True)
class ClassScore:
    """The one-vs-rest scores for a single class."""

    label: str
    base: float
    brier: float
    index: float
    adjusted: float


@dataclass(frozen=True)
class Report:
    """Classification metrics, coverage and per-class scores."""

    dataset: str
    n: int
    soft: bool
    accuracy: float
    macro_f1: float
    brier: float
    macro_index: float
    macro_adjusted: float
    classes: list[ClassScore] = field(default_factory=list)
    # records with a usable prediction, and the ones without
    scored: int = 0
    missing: int = 0

    @property
    def coverage(self) -> float:
        """Return the share of records that produced a usable prediction."""
        return self.scored / self.n if self.n else 0.0

    @property
    def summary(self) -> list[tuple[str, str, str]]:
        """The headline metrics as (name, value, gloss) rows."""
        return [
            ("accuracy", f"{self.accuracy:.3f}", "every record, missing ones wrong"),
            (
                "coverage",
                f"{self.coverage:.3f}",
                f"{self.scored}/{self.n} records produced a prediction"
                + (f", {self.missing} did not" if self.missing else ""),
            ),
            ("macro-F1", f"{self.macro_f1:.3f}", ""),
            (
                "Brier score",
                _cell(self.brier, "{:.3f}"),
                f"multiclass, lower is better, over the {self.scored} scored",
            ),
            (
                "Brier Index",
                _pct(self.macro_index),
                "macro one-vs-rest, higher is better; 50%-base-rate anchor",
            ),
            (
                "Adjusted Index",
                _pct(self.macro_adjusted),
                "macro, referenced to each class's own base rate",
            ),
        ]


def _pct(value: float) -> str:
    """Format an index as a percentage, or n/a where the base rate leaves it undefined."""
    return "n/a" if math.isnan(value) else f"{value:.1f}%"


def _cell(value: float, fmt: str) -> str:
    """Format one table cell, or n/a where the base rate leaves the value undefined."""
    return "n/a" if math.isnan(value) else fmt.format(value)


def has_forecast(r: dict) -> bool:
    """Return whether the record has a valid probability distribution or category."""
    if r.get("probs"):
        return True
    return bool(r.get("pred"))


def _prob(r: dict, label: str) -> float:
    """Return the predicted probability for one class, soft if present, else one-hot."""
    if probs := r.get("probs"):
        return float(probs.get(label, 0.0))
    return 1.0 if r["pred"] == label else 0.0


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


def _class_score(preds: list[dict], label: str) -> ClassScore:
    """Calculate one-vs-rest Brier metrics for a class.

    Calculate the class frequency from the supplied subset of valid forecasts.
    """
    n = len(preds)
    base = sum(r["label"] == label for r in preds) / n
    brier = sum((_prob(r, label) - float(r["label"] == label)) ** 2 for r in preds) / n
    spread = base * (1 - base)
    return ClassScore(
        label=label,
        base=base,
        brier=brier,
        index=100 * (1 - math.sqrt(brier)),
        adjusted=100 - 50 * math.sqrt(brier / spread) if spread else math.nan,
    )


def score(dataset: str, preds: list[dict], labels: list[str]) -> Report:
    """Calculate accuracy and macro-F1 over all records.

    Calculate Brier metrics over valid forecasts and report their coverage.
    """
    scored = [r for r in preds if has_forecast(r)]
    classes = [_class_score(scored, label) for label in labels] if scored else []
    adjusted = [c.adjusted for c in classes if not math.isnan(c.adjusted)]
    return Report(
        dataset=dataset,
        n=len(preds),
        soft=any(r.get("probs") for r in preds),
        accuracy=sum(r.get("pred") == r["label"] for r in preds) / len(preds),
        macro_f1=_macro_f1(preds, labels),
        brier=sum(c.brier for c in classes) if classes else math.nan,
        macro_index=sum(c.index for c in classes) / len(classes)
        if classes
        else math.nan,
        macro_adjusted=sum(adjusted) / len(adjusted) if adjusted else math.nan,
        classes=classes,
        scored=len(scored),
        missing=len(preds) - len(scored),
    )


def render(report: Report) -> str:
    """Render the report as the console table."""
    kind = "soft" if report.soft else "hard"
    header = (
        f"dataset: {report.dataset} "
        f"(n={report.n}, K={len(report.classes)} classes, {kind} predictions)"
    )
    lines = [header, ""]

    name_width = max(len(name) for name, _, _ in report.summary)
    for name, value, gloss in report.summary:
        row = f"  {name:<{name_width}}  {value:>7}"
        lines.append(f"{row}  ({gloss})" if gloss else row)
    if not report.soft:
        lines.append(
            "\nnote: hard predictions -> Brier is 2*error; "
            "add --probs to classify for a calibrated Brier"
        )

    if not report.classes:
        lines.append(
            "\nno record produced a prediction, so there is nothing to score per class"
        )
        return "\n".join(lines)

    label_width = max(len(c.label) for c in report.classes)
    headings = "".join(f"{heading:>10}" for heading, _, _ in CLASS_COLUMNS)
    lines += ["", f"  {'per class':<{label_width}}{headings}"]
    for c in report.classes:
        cells = "".join(
            f"{_cell(getattr(c, name), fmt):>10}" for _, name, fmt in CLASS_COLUMNS
        )
        lines.append(f"  {c.label:<{label_width}}{cells}")
    return "\n".join(lines)


def main() -> None:
    """Print the classification metrics for a scored dataset."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="conversations")
    p.add_argument("--data", type=Path, default=Path("data/generated"))
    a = p.parse_args()

    predictions = a.data / f"{a.dataset}.predictions.jsonl"
    preds = [
        json.loads(line)
        for line in predictions.read_text().splitlines()
        if line.strip()
    ]
    if not preds:
        # A round that died partway through leaves the file behind with nothing
        # in it, and every metric here divides by the number of predictions.
        raise SystemExit(
            f"{predictions} has no predictions in it. Run the router over the "
            "dataset first: make classify, or make decide for the decision model."
        )
    labels = json.loads((a.data / f"{a.dataset}.labels.json").read_text())
    print(render(score(a.dataset, preds, labels)))


if __name__ == "__main__":
    main()
