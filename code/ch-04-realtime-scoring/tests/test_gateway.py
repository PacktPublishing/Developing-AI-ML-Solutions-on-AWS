"""The gateway's decision rules, exercised without a network or an AWS account.

Covers the ensemble, the shadow split, and what happens when a model does not answer.
No network and no AWS: conftest sets ENDPOINTS before app is imported, and each test
substitutes its own scoring function, so the rules are exercised on their own.
"""

import app as gw
import httpx
import pytest

CLEAN = {
    "application_id": "A1",
    "age": 35,
    "monthly_income": 5000.0,
    "requested_amount": 10000.0,
    "dti": 0.2,
    "utilization": 0.3,
    "days_past_due": 0,
    "kyc_passed": True,
}


@pytest.fixture(autouse=True)
def _reset():
    """Every test starts from the shipped defaults, whatever the previous one set."""
    shadow, once = gw.SHADOW_MODELS, gw._once
    gw.SHADOW_MODELS = set()
    yield
    gw.SHADOW_MODELS, gw._once = shadow, once


def answers(**pds):
    """Substitute a scoring function; a value may be an exception to raise instead."""

    def _once(target, body):
        name = "scorecard" if target.endswith("a") else "challenger"
        value = pds[name]
        if isinstance(value, Exception):
            raise value
        return value

    gw._once = _once


def decide(**overrides):
    """Decide one application, starting from a clean one."""
    return gw.decide(gw.Application(**{**CLEAN, **overrides}))


def test_ensemble_averages_every_deciding_model():
    """Every endpoint outside SHADOW_MODELS contributes equally to the PD."""
    answers(scorecard=0.05, challenger=0.15)
    out = decide()
    assert out.pd == pytest.approx(0.10)
    assert out.decision == "REFER"  # 0.10 is above the approve cutoff


def test_shadow_model_is_logged_but_does_not_move_the_decision():
    """A shadow model is measured and logged, but the champion alone decides."""
    gw.SHADOW_MODELS = {"challenger"}
    answers(scorecard=0.05, challenger=0.15)
    out = decide()
    assert out.pd == pytest.approx(0.05)  # the champion alone
    assert out.decision == "APPROVE"  # which the ensemble would not have given
    assert out.model_pds["challenger"] == pytest.approx(0.15)  # still measured


def test_unreachable_deciding_model_refers_instead_of_deciding():
    """A deciding model that never answers sends the application to a human."""
    answers(scorecard=httpx.ReadTimeout("slow"), challenger=0.05)
    out = decide()
    assert out.decision == "REFER"
    assert "MODEL_UNAVAILABLE" in out.reasons
    assert out.unavailable == ["scorecard"]


def test_unreachable_shadow_model_leaves_the_decision_alone():
    """A shadow model that fails costs nothing, since it was never in the number."""
    gw.SHADOW_MODELS = {"challenger"}
    answers(scorecard=0.05, challenger=httpx.ReadTimeout("slow"))
    out = decide()
    assert out.decision == "APPROVE"
    assert out.unavailable == ["challenger"]


def test_hard_rules_decline_without_needing_a_model():
    """A hard policy failure decides on its own terms, models up or down."""
    answers(scorecard=httpx.ReadTimeout("x"), challenger=httpx.ReadTimeout("x"))
    out = decide(kyc_passed=False)
    assert out.decision == "DECLINE"
    assert out.reasons == ["KYC_FAILED"]


def test_every_model_shadowed_is_a_misconfiguration_not_an_outage():
    """Nothing left to decide with is named apart from an endpoint outage."""
    gw.SHADOW_MODELS = {"scorecard", "challenger"}
    answers(scorecard=0.05, challenger=0.15)
    out = decide()
    assert out.decision == "REFER"
    assert "NO_DECIDING_MODEL" in out.reasons


def test_a_call_is_retried_before_the_model_is_given_up_on():
    """A model gets MODEL_ATTEMPTS tries before it counts as unavailable."""
    calls = {"n": 0}

    def _once(target, body):
        calls["n"] += 1
        if calls["n"] < gw.MODEL_ATTEMPTS:
            raise httpx.ReadTimeout("first attempt")
        return 0.05

    gw._once = _once
    gw.ENDPOINTS = {"scorecard": "http://a"}
    try:
        assert decide().decision == "APPROVE"
        assert calls["n"] == gw.MODEL_ATTEMPTS
    finally:
        gw.ENDPOINTS = {"scorecard": "http://a", "challenger": "http://b"}
