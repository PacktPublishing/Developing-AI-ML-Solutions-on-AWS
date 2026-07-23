"""The delegated-authority routing rule: the one piece that must never drift.

Pure unit tests, no Docker or cloud. They pin the tier boundaries and the
two-dimensional escalation, so a change to the thresholds, the role names, or
the returned keys fails loudly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "underwriting-agent"))

from authority import decide  # noqa: E402

# Mirrors AUTHORITY_TIERS in knowledge-base/corpus.py.
TIERS = [
    {
        "tier": 1,
        "name": "Underwriter",
        "max_exposure_usd": 2_000_000,
        "max_risk_profile": 6,
    },
    {
        "tier": 2,
        "name": "Managing director",
        "max_exposure_usd": 10_000_000,
        "max_risk_profile": 8,
    },
    {
        "tier": 3,
        "name": "Chief executive",
        "max_exposure_usd": None,
        "max_risk_profile": 10,
    },
]


def test_small_low_risk_stays_with_underwriter():
    decision = decide(1_500_000, 5, TIERS)
    assert decision["role"] == "Underwriter"
    assert decision["escalates"] is False


def test_small_but_risky_escalates_on_risk_alone():
    # under the underwriter's 2 million ceiling, but profile 7 exceeds their 6
    decision = decide(900_000, 7, TIERS)
    assert decision["role"] == "Managing director"
    assert decision["escalates"] is True


def test_large_exposure_goes_to_chief_executive():
    # 12 million exceeds the managing director's 10 million ceiling
    decision = decide(12_000_000, 8, TIERS)
    assert decision["role"] == "Chief executive"
    assert decision["escalates"] is True


def test_top_tier_has_no_exposure_ceiling():
    decision = decide(500_000_000, 10, TIERS)
    assert decision["role"] == "Chief executive"
    assert decision["tier"] == 3
