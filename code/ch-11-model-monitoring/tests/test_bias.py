"""Hold src/bias.py to smclarify's numbers, metric by metric.

smclarify is AWS's open-source implementation of the same metrics, Apache-2.0,
and it is the reference this chapter's own implementation is checked against. It
is a test dependency only: nothing the chapter runs imports it.

This is the point of the exercise. AWS's guidance for replacing Clarify says to
compute the metrics yourself because they are standardized arithmetic, and the
way to earn that claim is to show the arithmetic agreeing with the reference on
data that includes the awkward cases, not to assert it.
"""

import numpy as np
import pandas as pd
import pytest
from bias import POST_TRAINING, PRE_TRAINING, counts_for

smclarify_metrics = pytest.importorskip(
    "smclarify.bias.metrics",
    reason="smclarify is the reference these metrics are compared against",
)


def frames():
    """Datasets to compare on, including the ones that divide by zero."""
    rng = np.random.default_rng(11)
    n = 400
    yield (
        "random",
        pd.DataFrame(
            {
                "facet": rng.integers(0, 2, n),
                "label": rng.integers(0, 2, n),
                "predicted": rng.integers(0, 2, n),
            }
        ),
    )
    # A model that is right about one facet and wrong about the other, which is
    # the shape every one of these metrics exists to notice.
    facet = np.repeat([0, 1], n // 2)
    label = np.tile([0, 1], n // 2)
    predicted = np.where(facet == 1, label, 1 - label)
    yield (
        "skewed",
        pd.DataFrame({"facet": facet, "label": label, "predicted": predicted}),
    )
    # Every prediction positive, so the negatives' denominators are zero.
    yield (
        "all positive",
        pd.DataFrame(
            {
                "facet": np.repeat([0, 1], n // 2),
                "label": np.tile([0, 1], n // 2),
                "predicted": np.ones(n, dtype=int),
            }
        ),
    )
    # One facet is tiny, which is the realistic awkward case.
    small = np.zeros(n, dtype=int)
    small[:5] = 1
    yield (
        "tiny facet",
        pd.DataFrame(
            {
                "facet": small,
                "label": np.tile([0, 1], n // 2),
                "predicted": np.tile([1, 0], n // 2),
            }
        ),
    )


def as_series(frame):
    """Build the masks smclarify takes: the facet and the positive-label ones."""
    return (
        frame["facet"],
        frame["facet"] == 1,
        frame["label"] == 1,
        frame["predicted"] == 1,
    )


# The reference's metrics do not all take the same arguments: CI reads the facet
# alone, DPL the labels, DPPL and DI the predictions, and the rest need both.
def reference(metric, frame):
    """Call smclarify's version of one metric with the arguments it takes."""
    feature, facet, label, predicted = as_series(frame)
    fn = getattr(smclarify_metrics, metric)
    if metric == "CI":
        return fn(feature, facet)
    if metric == "DPL":
        return fn(feature, facet, label)
    if metric in ("DPPL", "DI"):
        return fn(feature, facet, predicted)
    return fn(feature, facet, label, predicted)


@pytest.mark.parametrize("name,frame", list(frames()), ids=lambda v: getattr(v, "", v))
@pytest.mark.parametrize("metric", sorted(PRE_TRAINING))
def test_pre_training_matches_smclarify(metric, name, frame):
    """Each pre-training metric equals smclarify's value for it."""
    _, facet, _, _ = as_series(frame)
    expected = reference(metric, frame)

    a = counts_for(~facet, frame["label"])
    d = counts_for(facet, frame["label"])
    actual = PRE_TRAINING[metric](a, d)

    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), (
        f"{metric} on {name}: ours {actual}, smclarify {expected}"
    )


@pytest.mark.parametrize("name,frame", list(frames()), ids=lambda v: getattr(v, "", v))
@pytest.mark.parametrize("metric", sorted(POST_TRAINING))
def test_post_training_matches_smclarify(metric, name, frame):
    """Each post-training metric equals smclarify's value for it."""
    _, facet, _, _ = as_series(frame)
    expected = reference(metric, frame)

    a = counts_for(~facet, frame["label"], frame["predicted"])
    d = counts_for(facet, frame["label"], frame["predicted"])
    actual = POST_TRAINING[metric](a, d)

    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), (
        f"{metric} on {name}: ours {actual}, smclarify {expected}"
    )
