"""The scoring maths in evaluate.py, against values worked out by hand.

The report is what the chapter reads to decide whether a classifier is good enough,
so every number in it is checked here rather than eyeballed in the console table.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate

LABELS = ["a", "b"]


def test_perfect_hard_predictions_score_100():
    """A classifier that is always right has Brier 0, so both indices reach 100."""
    preds = [{"pred": "a", "label": "a"}, {"pred": "b", "label": "b"}]
    report = evaluate.score("perfect", preds, LABELS)

    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.brier == 0.0
    assert report.macro_index == 100.0
    assert report.macro_adjusted == 100.0
    assert report.soft is False


def test_soft_probabilities_are_scored_not_the_argmax():
    """A confident-but-hedged prediction costs Brier even when the argmax is right.

    Both rows are classified correctly, so accuracy is 1.0 while the index is not:
    that gap is the whole reason the chapter reports Brier alongside accuracy.
    """
    preds = [
        {"pred": "a", "label": "a", "probs": {"a": 0.75, "b": 0.25}},
        {"pred": "b", "label": "b", "probs": {"a": 0.25, "b": 0.75}},
    ]
    report = evaluate.score("soft", preds, LABELS)
    a = next(c for c in report.classes if c.label == "a")

    assert report.accuracy == 1.0
    assert report.soft is True
    # each row contributes (0.75 - 1)^2 = 0.0625, so the mean is 0.0625
    assert a.brier == 0.0625
    assert a.index == 75.0
    # base 0.5 -> spread 0.25; 100 - 50 * sqrt(0.0625 / 0.25) = 75
    assert a.adjusted == 75.0


def test_a_single_class_dataset_has_no_adjusted_index():
    """With one class present the base rate leaves no spread to adjust against."""
    preds = [{"pred": "a", "label": "a"}, {"pred": "a", "label": "a"}]
    report = evaluate.score("degenerate", preds, LABELS)
    a = next(c for c in report.classes if c.label == "a")

    assert a.base == 1.0
    # both classes are degenerate here (one is always present, the other never),
    # so there is nothing left to average and the macro is NaN too
    assert math.isnan(a.adjusted)
    assert math.isnan(report.macro_adjusted)
    # NaN reaches the table as n/a rather than as the string "nan"
    rendered = evaluate.render(report)
    assert "n/a" in rendered
    assert "nan" not in rendered


def test_the_macro_average_skips_a_degenerate_class():
    """A class nobody has skips the adjusted average instead of poisoning it."""
    preds = [{"pred": "a", "label": "a"}, {"pred": "b", "label": "b"}]
    report = evaluate.score("absent-class", preds, ["a", "b", "c"])
    c = next(k for k in report.classes if k.label == "c")

    assert c.base == 0.0
    assert math.isnan(c.adjusted)
    # a and b are both perfect, so the average over the two real classes is 100
    assert report.macro_adjusted == 100.0


def test_probability_lookup_is_per_label():
    """A row without probs falls back to one-hot on its own prediction."""
    hard = {"pred": "a", "label": "b"}
    assert evaluate._prob(hard, "a") == 1.0
    assert evaluate._prob(hard, "b") == 0.0

    soft = {"pred": "a", "label": "a", "probs": {"a": 0.6}}
    assert evaluate._prob(soft, "a") == 0.6
    # a label the model never scored reads as zero, not as a missing key
    assert evaluate._prob(soft, "b") == 0.0


def test_macro_f1_penalises_a_class_that_is_never_predicted():
    """Predicting one class for everything gives that class F1 1.0 and the other 0."""
    preds = [
        {"pred": "a", "label": "a"},
        {"pred": "a", "label": "b"},
    ]
    report = evaluate.score("collapsed", preds, LABELS)

    assert report.accuracy == 0.5
    # a: precision 0.5, recall 1.0 -> F1 2/3; b: never predicted -> F1 0
    assert report.macro_f1 == ((2 * 0.5 * 1.0 / 1.5) + 0.0) / 2
