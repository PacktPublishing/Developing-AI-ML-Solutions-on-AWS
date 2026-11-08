"""Bias metrics computed from label counts and confusion-matrix counts.

SageMaker Clarify is closed to new customers, and AWS's own guidance for
replacing it says the metrics are the part worth keeping: "the bias metrics are
standardized arithmetic, not a library dependency", to be reproduced with pandas
and scikit-learn as code you own, test and version. This module is that code.

Every function takes a facet mask and label masks, which is the shape Clarify
itself uses: a facet splits the population into the favored group a and the
disfavored group d, and the metrics compare the two.

tests/test_bias.py checks each one against smclarify, AWS's open-source
implementation of the same metrics (Apache-2.0), so the numbers here are held to
the reference rather than asserted to match it.

The interpretation of each value is in the chapter; a metric near zero means the
facets look alike by that measure, and the sign says which facet is favored.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Clarify reports infinity where a ratio's denominator is zero rather than
# failing, and smclarify uses this value for it.
INFINITY = float("inf")


def _rate(numerator: float, denominator: float) -> float:
    """Divide, reporting infinity on a zero denominator as Clarify does."""
    if denominator == 0:
        return INFINITY if numerator > 0 else 0.0
    return numerator / denominator


@dataclass(frozen=True)
class Counts:
    """The counts every metric below is built from, for one facet."""

    n: int
    """every row in the facet"""
    positives: int
    """rows the data labels positive"""
    predicted_positives: int
    """rows the model predicts positive"""
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def negatives(self) -> int:
        """Rows the data labels negative."""
        return self.n - self.positives

    @property
    def predicted_negatives(self) -> int:
        """Rows the model predicts negative."""
        return self.n - self.predicted_positives


def counts_for(
    facet: pd.Series, label: pd.Series, predicted: pd.Series | None = None
) -> Counts:
    """Count one facet's labels and, when given predictions, its outcomes."""
    facet = facet.astype(bool).to_numpy()
    y = label.astype(bool).to_numpy()[facet]
    if predicted is None:
        yhat = np.zeros_like(y)
    else:
        yhat = predicted.astype(bool).to_numpy()[facet]
    return Counts(
        n=int(facet.sum()),
        positives=int(y.sum()),
        predicted_positives=int(yhat.sum()),
        true_positives=int((y & yhat).sum()),
        false_positives=int((~y & yhat).sum()),
        true_negatives=int((~y & ~yhat).sum()),
        false_negatives=int((y & ~yhat).sum()),
    )


# Pre-training: these read the data only, so they hold for any model.


def class_imbalance(a: Counts, d: Counts) -> float:
    """CI: how unevenly the facets are represented at all. Range [-1, +1]."""
    return _rate(a.n - d.n, a.n + d.n)


def difference_in_proportions_of_labels(a: Counts, d: Counts) -> float:
    """DPL: the gap in the rate of positive labels. Range [-1, +1]."""
    return _rate(a.positives, a.n) - _rate(d.positives, d.n)


# Post-training: these compare predictions, so they describe a given model.


def difference_in_positive_proportions(a: Counts, d: Counts) -> float:
    """DPPL: the gap in the rate of positive predictions. Range [-1, +1]."""
    return _rate(a.predicted_positives, a.n) - _rate(d.predicted_positives, d.n)


def disparate_impact(a: Counts, d: Counts) -> float:
    """DI: the same comparison as a ratio. 1 is parity, range [0, inf)."""
    return _rate(_rate(d.predicted_positives, d.n), _rate(a.predicted_positives, a.n))


def recall_difference(a: Counts, d: Counts) -> float:
    """RD: the gap in recall, so in positives the model does not miss."""
    recall_a = _rate(a.true_positives, a.true_positives + a.false_negatives)
    recall_d = _rate(d.true_positives, d.true_positives + d.false_negatives)
    return recall_a - recall_d


def accuracy_difference(a: Counts, d: Counts) -> float:
    """AD: the gap in accuracy between the facets."""
    correct_a = _rate(a.true_positives + a.true_negatives, a.n)
    correct_d = _rate(d.true_positives + d.true_negatives, d.n)
    return correct_a - correct_d


def specificity_difference(a: Counts, d: Counts) -> float:
    """SD: the gap in specificity, so in negatives the model gets right.

    This one is d - a, unlike the differences above. DRR and TE run the same
    way round. The signs are not a house style to be made consistent: they are
    the reference's, and tests/test_bias.py is what keeps them honest.
    """
    spec_a = _rate(a.true_negatives, a.true_negatives + a.false_positives)
    spec_d = _rate(d.true_negatives, d.true_negatives + d.false_positives)
    return spec_d - spec_a


def difference_in_acceptance_rates(a: Counts, d: Counts) -> float:
    """DAR: the gap in precision among the rows predicted positive."""
    precision_a = _rate(a.true_positives, a.true_positives + a.false_positives)
    precision_d = _rate(d.true_positives, d.true_positives + d.false_positives)
    return precision_a - precision_d


def difference_in_rejection_rates(a: Counts, d: Counts) -> float:
    """DRR: the gap in precision among the rows predicted negative."""
    rejection_d = _rate(d.true_negatives, d.true_negatives + d.false_negatives)
    rejection_a = _rate(a.true_negatives, a.true_negatives + a.false_negatives)
    return rejection_d - rejection_a


def treatment_equality(a: Counts, d: Counts) -> float:
    """TE: the gap in how the two error kinds trade off against each other."""
    ratio_d = _rate(d.false_negatives, d.false_positives)
    ratio_a = _rate(a.false_negatives, a.false_positives)
    if ratio_d == INFINITY and ratio_a == INFINITY:
        return 0.0
    return ratio_d - ratio_a


PRE_TRAINING = {
    "CI": class_imbalance,
    "DPL": difference_in_proportions_of_labels,
}
POST_TRAINING = {
    "DPPL": difference_in_positive_proportions,
    "DI": disparate_impact,
    "RD": recall_difference,
    "AD": accuracy_difference,
    "SD": specificity_difference,
    "DAR": difference_in_acceptance_rates,
    "DRR": difference_in_rejection_rates,
    "TE": treatment_equality,
}


@dataclass
class Report:
    """One facet's metrics, ready to log as a run's parameters and metrics."""

    facet: str
    """the column the facet was taken from"""
    value: str
    """the value of that column which marks the disfavored facet d"""
    n_favored: int
    n_disfavored: int
    metrics: dict[str, float] = field(default_factory=dict)


def report_for(
    frame: pd.DataFrame,
    facet_column: str,
    disfavored_value: object,
    label_column: str,
    predicted_column: str | None = None,
) -> Report:
    """Compute the metrics for one facet of one frame.

    The disfavored facet d is where facet_column equals disfavored_value, and
    everything else is the favored facet a. Passing no predicted_column computes
    the pre-training metrics only, which is the check worth running before any
    model exists.
    """
    disfavored = frame[facet_column] == disfavored_value
    label = frame[label_column]
    predicted = frame[predicted_column] if predicted_column else None

    a = counts_for(~disfavored, label, predicted)
    d = counts_for(disfavored, label, predicted)

    wanted = dict(PRE_TRAINING)
    if predicted_column:
        wanted.update(POST_TRAINING)

    return Report(
        facet=facet_column,
        value=str(disfavored_value),
        n_favored=a.n,
        n_disfavored=d.n,
        metrics={name: float(fn(a, d)) for name, fn in wanted.items()},
    )


def parse_facet(spec: str) -> tuple[str, str]:
    """Split a "column=value" facet into its parts."""
    if "=" not in spec:
        raise ValueError(
            f"a facet is written column=value, which {spec!r} is not. A numeric "
            "column has to be banded into a categorical one first, the way "
            "Clarify's own age example does."
        )
    column, value = spec.split("=", 1)
    return column.strip(), value.strip()


def reports_for(
    frame: pd.DataFrame,
    facets: list[str],
    label_column: str | None,
    predicted_column: str | None = None,
) -> list[Report]:
    """Compute every facet's metrics for one frame.

    Captured traffic carries the request and the score and no outcome, so there
    is no label column for it. CI, DPPL and DI are the metrics that survive
    that: they read the facet and the predictions, never the ground truth. The
    rest need outcomes, and on a live endpoint those arrive later, if at all.
    """
    out = []
    for spec in facets:
        column, value = parse_facet(spec)
        if column not in frame.columns:
            continue
        disfavored = (frame[column] == value).sum()
        if disfavored == 0 or disfavored == len(frame):
            # Every metric here divides by one facet's size, so a facet that
            # matches no rows, or all of them, has no metrics rather than
            # extreme ones. A misspelled value looks exactly like this, and
            # printing numbers for it would hide the mistake.
            out.append(
                Report(
                    facet=column,
                    value=str(value),
                    n_favored=len(frame) - int(disfavored),
                    n_disfavored=int(disfavored),
                )
            )
            continue
        if label_column is None:
            # Stand in a constant label so the counts are well defined; only the
            # metrics that ignore it are reported back.
            frame = frame.assign(_no_label=0)
            report = report_for(frame, column, value, "_no_label", predicted_column)
            report.metrics = {
                k: v for k, v in report.metrics.items() if k in ("CI", "DPPL", "DI")
            }
        else:
            report = report_for(frame, column, value, label_column, predicted_column)
        out.append(report)
    return out
