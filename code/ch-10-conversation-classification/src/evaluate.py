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

A prediction that is missing is not a prediction of nothing. A record the job
could not run, or whose answer could not be read, has no distribution at all, and
scoring it as an all-zero vector flatters it: an all-zero vector is not a
probability distribution, and it can score better than a wrong answer that was
at least a forecast. So the probability metrics are computed over the records
that produced a usable forecast, the report says how many that was, and accuracy
counts every record, missing ones included, because a router that answers
nothing routed nothing.

Usage:
  uv run src/evaluate.py --dataset conversations
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# The per-class table, as (heading, ClassScore field, format). One place to edit,
# so a heading can never drift away from the column under it.
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
    """Everything evaluate.py knows about one scored dataset."""

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
    """Report whether this record produced something that can be scored.

    A soft prediction needs a distribution; a hard one needs a label. A record
    with neither failed, and a failure has no probability attached to it.
    """
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
    """Return the one-vs-rest scores for one class: base rate, Brier, and both indices.

    `preds` here is the scored subset, the records that produced a forecast. The
    base rate is that subset's own, so the adjusted index references the problem
    the numbers actually describe.
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
    """Score the predictions against the label set.

    Accuracy and macro-F1 are over every record, so a record with no prediction
    counts against the run. The Brier score and both indices are over the
    records that produced a forecast, because the others have no distribution to
    score, and the report carries how many of each there were.
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

    preds = [
        json.loads(line)
        for line in (a.data / f"{a.dataset}.predictions.jsonl").read_text().splitlines()
    ]
    labels = json.loads((a.data / f"{a.dataset}.labels.json").read_text())
    print(render(score(a.dataset, preds, labels)))


if __name__ == "__main__":
    main()
