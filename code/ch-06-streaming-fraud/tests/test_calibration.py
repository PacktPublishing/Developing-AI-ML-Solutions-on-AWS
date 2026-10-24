"""The score scale and the bands the consumer runs on.

A cut's false-alarm rate is the claim this chapter makes to a fraud desk, so the
tests check it against measured rates on legitimate traffic the calibrator never
saw, not only against the formula that produced it.
"""

import numpy as np
import pytest

from calibration import BLOCK_CUT, GAMMA, REVIEW_CUT, decide, to_fpr, to_score


@pytest.fixture(scope="module")
def reference():
    """Return a sorted reference of legitimate probabilities, as the calibrator holds it."""
    rng = np.random.default_rng(6)
    return np.sort(rng.beta(1.2, 30.0, size=20_000))


@pytest.fixture(scope="module")
def fresh_legit():
    """Return legitimate probabilities from the same distribution, unseen by the reference."""
    rng = np.random.default_rng(7)
    return rng.beta(1.2, 30.0, size=20_000)


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------
def test_scores_stay_inside_the_published_range(reference, fresh_legit):
    """The endpoint publishes a 0 to 1000 integer, so nothing may fall outside it."""
    score = to_score(fresh_legit, reference)
    assert score.min() >= 0
    assert score.max() <= 1000
    assert score.dtype.kind == "i"


def test_score_is_monotone_in_the_probability(reference):
    """A ranking the model produces must survive the rescaling intact."""
    probabilities = np.linspace(0.0, 1.0, 500)
    score = to_score(probabilities, reference)
    assert np.all(np.diff(score) >= 0)


def test_a_higher_probability_never_scores_lower(reference):
    """The cheapest possible statement of the same property."""
    assert to_score([0.9], reference)[0] >= to_score([0.1], reference)[0]


# ---------------------------------------------------------------------------
# The false-alarm claim
# ---------------------------------------------------------------------------
def test_gamma_puts_a_two_percent_false_alarm_rate_at_900():
    """The exponent is solved from one anchor, and 900 is where it lands."""
    assert to_fpr(900) == pytest.approx(0.02, abs=1e-6)
    assert GAMMA == pytest.approx(np.log(0.9) / np.log(0.98))


@pytest.mark.parametrize("cut", [500, 700, 900, 955, 990])
def test_claimed_rate_holds_on_legitimates_the_reference_never_saw(
    cut, reference, fresh_legit
):
    """Calibrating on held-out rows is what makes claimed and measured agree."""
    score = to_score(fresh_legit, reference)
    measured = (score >= cut).mean()
    assert measured == pytest.approx(to_fpr(cut), abs=0.01)


def test_to_fpr_falls_as_the_cut_rises():
    """A higher cut cannot wrongly stop more good customers than a lower one."""
    cuts = np.array([100, 300, 500, 700, 900, 1000])
    assert np.all(np.diff(to_fpr(cuts)) <= 0)


def test_a_cut_of_1000_claims_no_false_alarms():
    """The top of the scale is the degenerate case the formula must still handle."""
    assert to_fpr(1000) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# The bands
# ---------------------------------------------------------------------------
def test_the_three_outcomes_are_the_ones_the_consumer_writes():
    """These strings reach DynamoDB, the decisions stream and the dashboard."""
    seen = {decide(s, REVIEW_CUT, BLOCK_CUT) for s in range(0, 1001, 5)}
    assert seen == {"approve", "investigate", "block"}


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "approve"),
        (REVIEW_CUT - 1, "approve"),
        (REVIEW_CUT, "investigate"),
        (BLOCK_CUT - 1, "investigate"),
        (BLOCK_CUT, "block"),
        (1000, "block"),
    ],
)
def test_the_cuts_are_inclusive_lower_bounds(score, expected):
    """A score exactly on a cut belongs to the band above it."""
    assert decide(score, REVIEW_CUT, BLOCK_CUT) == expected


def test_the_block_cut_is_checked_before_the_review_cut():
    """Ordered, first match wins: a score above both is a block, never a review."""
    assert decide(990, review_cut=700, block_cut=900) == "block"


def test_bands_do_not_overlap_and_leave_no_gap(reference, fresh_legit):
    """Every score falls in exactly one band, and inside that band's range."""
    score = to_score(fresh_legit, reference)
    outcome = np.array([decide(s, REVIEW_CUT, BLOCK_CUT) for s in score])
    assert len(outcome) == len(score)
    for band, lo, hi in [
        ("approve", 0, REVIEW_CUT),
        ("investigate", REVIEW_CUT, BLOCK_CUT),
        ("block", BLOCK_CUT, 1001),
    ]:
        in_band = score[outcome == band]
        if in_band.size:
            assert in_band.min() >= lo
            assert in_band.max() < hi


def test_the_review_cut_sits_below_the_block_cut():
    """Reversing them would make the middle band unreachable."""
    assert REVIEW_CUT < BLOCK_CUT


# ---------------------------------------------------------------------------
# The contract the endpoint and the consumer share
# ---------------------------------------------------------------------------
def test_a_threshold_carried_onto_the_scale_keeps_its_meaning(reference, fresh_legit):
    """Converting a probability threshold to a score must not change who is blocked."""
    probability_threshold = 0.1716641054193909
    block_cut = int(to_score([probability_threshold], reference)[0])
    by_probability = fresh_legit >= probability_threshold
    by_score = to_score(fresh_legit, reference) >= block_cut
    disagreement = (by_probability != by_score).mean()
    assert disagreement < 0.005
