"""The limit rules, and the two copies of them that must not drift.

decide() is the chapter's whole credit policy in six lines, so the thresholds and
the order they are checked in are worth pinning. The same function exists twice,
once for the step that runs as a script and once for the Lambda, and a rule
changed in one copy and not the other would move limits on real accounts.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from decide import (
    DECREASE_PD,
    DECREASE_STEP,
    INCREASE_PD,
    INCREASE_STEP,
    INCREASE_UTILIZATION,
    decide,
)

ROOT = Path(__file__).resolve().parents[1]


def row(pd_12m: float, utilization: float, current_limit: float = 1000.0) -> pd.Series:
    """One account as the step sees it."""
    return pd.Series(
        {"pd_12m": pd_12m, "utilization": utilization, "current_limit": current_limit}
    )


# ---------------------------------------------------------------------------
# The three outcomes
# ---------------------------------------------------------------------------
def test_low_risk_and_a_hard_working_limit_earns_an_increase():
    """Both conditions met: the limit goes up by the documented step."""
    action, limit = decide(row(pd_12m=0.01, utilization=0.9, current_limit=1000))
    assert action == "INCREASE"
    assert limit == pytest.approx(1200)


def test_high_risk_loses_limit():
    """Risk above the decrease cut costs the account 30 percent of its limit."""
    action, limit = decide(row(pd_12m=0.40, utilization=0.1, current_limit=1000))
    assert action == "DECREASE"
    assert limit == pytest.approx(700)


def test_everything_else_is_left_alone():
    """Neither rule fires, so the limit is returned unchanged."""
    action, limit = decide(row(pd_12m=0.10, utilization=0.2, current_limit=1000))
    assert action == "KEEP"
    assert limit == pytest.approx(1000)


def test_low_risk_on_an_idle_limit_is_not_an_increase():
    """Both conditions are required: cheap risk alone does not earn more limit."""
    action, _ = decide(row(pd_12m=0.01, utilization=0.1))
    assert action == "KEEP"


# ---------------------------------------------------------------------------
# The boundaries, which are where a rule change shows up first
# ---------------------------------------------------------------------------
def test_the_increase_thresholds_are_strict():
    """At exactly the cut-offs the account does not qualify."""
    assert decide(row(pd_12m=INCREASE_PD, utilization=0.9))[0] == "KEEP"
    assert decide(row(pd_12m=0.01, utilization=INCREASE_UTILIZATION))[0] == "KEEP"


def test_the_decrease_threshold_is_strict():
    """At exactly the cut the account keeps its limit; a hair above, it does not."""
    assert decide(row(pd_12m=DECREASE_PD, utilization=0.1))[0] == "KEEP"
    assert decide(row(pd_12m=DECREASE_PD + 0.001, utilization=0.1))[0] == "DECREASE"


def test_risk_is_checked_before_utilization():
    """A risky account with a hard-working limit loses limit rather than gaining it."""
    action, _ = decide(row(pd_12m=0.40, utilization=0.95))
    assert action == "DECREASE"


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("limit", [1000, 2500, 333, 1_000_000])
def test_a_new_limit_is_rounded_to_the_nearest_ten(limit):
    """Limits are published in tens; a customer never sees 1199.7."""
    for account in (row(0.01, 0.9, limit), row(0.40, 0.1, limit)):
        _, new_limit = decide(account)
        assert new_limit % 10 == 0


def test_the_steps_are_the_documented_percentages():
    """The new limits are the constants applied, not numbers drifted in by hand."""
    _, up = decide(row(0.01, 0.9, 1000))
    _, down = decide(row(0.40, 0.1, 1000))
    assert up == pytest.approx(round(1000 * (1 + INCREASE_STEP), -1))
    assert down == pytest.approx(round(1000 * (1 - DECREASE_STEP), -1))


def test_a_kept_limit_is_returned_untouched():
    """KEEP must not round, or a schedule of no-op runs would drift the limit."""
    _, limit = decide(row(0.10, 0.2, current_limit=1234.56))
    assert limit == pytest.approx(1234.56)


# ---------------------------------------------------------------------------
# The two copies
# ---------------------------------------------------------------------------
def _load_lambda_copy():
    """Import src/pipeline/scripts/decide.py under its own name."""
    path = ROOT / "src" / "pipeline" / "scripts" / "decide.py"
    spec = importlib.util.spec_from_file_location("pipeline_decide", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_lambda_copy_uses_the_same_thresholds():
    """Five constants, duplicated in two files, that must stay equal."""
    other = _load_lambda_copy()
    assert (
        other.INCREASE_PD,
        other.INCREASE_UTILIZATION,
        other.INCREASE_STEP,
        other.DECREASE_PD,
        other.DECREASE_STEP,
    ) == (
        INCREASE_PD,
        INCREASE_UTILIZATION,
        INCREASE_STEP,
        DECREASE_PD,
        DECREASE_STEP,
    )


@pytest.mark.parametrize(
    "pd_12m,utilization",
    [(0.01, 0.9), (0.40, 0.1), (0.10, 0.2), (0.05, 0.9), (0.25, 0.1), (0.049, 0.601)],
)
def test_the_two_copies_decide_alike(pd_12m, utilization):
    """The same account must get the same answer from either implementation."""
    other = _load_lambda_copy()
    account = row(pd_12m, utilization, 1000)
    assert decide(account) == other.decide(account)
