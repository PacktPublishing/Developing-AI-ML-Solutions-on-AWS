"""The delegated-authority routing rule: the one piece that must never drift."""

from authority import decide

# Mirrors AUTHORITY_TIERS in src/knowledge-base/corpus.py.
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
    """A small, low-risk loan stays with the underwriter."""
    decision = decide(1_500_000, 5, TIERS)
    assert decision["role"] == "Underwriter"
    assert decision["escalates"] is False


def test_small_but_risky_escalates_on_risk_alone():
    """A small loan with a high risk profile escalates on risk alone."""
    # under the underwriter's 2 million ceiling, but profile 7 exceeds their 6
    decision = decide(900_000, 7, TIERS)
    assert decision["role"] == "Managing director"
    assert decision["escalates"] is True


def test_large_exposure_goes_to_chief_executive():
    """Exposure above the managing director's ceiling escalates to the chief executive."""
    # 12 million exceeds the managing director's 10 million ceiling
    decision = decide(12_000_000, 8, TIERS)
    assert decision["role"] == "Chief executive"
    assert decision["escalates"] is True


def test_top_tier_has_no_exposure_ceiling():
    """The top tier has no exposure ceiling."""
    decision = decide(500_000_000, 10, TIERS)
    assert decision["role"] == "Chief executive"
    assert decision["tier"] == 3
