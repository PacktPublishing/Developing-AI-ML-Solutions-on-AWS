"""The rollout: the agent's deterministic split, monotone widening, and router assignment."""

from appconfig import AgentAppConfig
from router import Router

LOANS = [f"L{i:06d}" for i in range(2000)]


def _variants(agent_url: str, profile: str) -> dict[str, str]:
    """Ask the agent for every loan's variant under one rollout profile."""
    client = AgentAppConfig(agent_url, "credit-governance", "local", profile)
    return {
        x: client.configuration({"loanId": x})["challenger_rollout"]["_variant"]
        for x in LOANS
    }


def test_split_share_matches_the_flag(agent_url):
    """About pct% of loans route to the challenger."""
    got = _variants(agent_url, "rollout")
    share = sum(v == "challenger" for v in got.values()) / len(LOANS)
    assert 0.17 < share < 0.23  # ~20%


def test_assignment_is_deterministic(agent_url):
    """A loan id always resolves to the same variant."""
    client = AgentAppConfig(agent_url, "credit-governance", "local", "rollout")
    seen = {
        client.configuration({"loanId": "L000042"})["challenger_rollout"]["_variant"]
        for _ in range(10)
    }
    assert len(seen) == 1


def test_widening_never_flips_a_loan_back(agent_url):
    """Bumping pct only adds loans to the challenger cohort; none leave it."""
    in20 = {x for x, v in _variants(agent_url, "rollout").items() if v == "challenger"}
    in50 = {
        x for x, v in _variants(agent_url, "rollout50").items() if v == "challenger"
    }
    assert in20 < in50  # strictly grows, and the 20% cohort stays in


def test_router_scores_with_the_picked_model(stub_flags):
    """The router routes each loan to the model the flag picks and decides on its pd."""
    calls = []

    def score(model_name, loan):
        calls.append((loan["loanId"], model_name))
        return 0.8 if model_name == "credit-challenger" else 0.1

    router = Router(stub_flags, score)
    out = [router.route({"loanId": f"L{i:06d}"}) for i in range(200)]

    # every routed model matches the loan's variant, and the pd drives the decision
    for d in out:
        expected = (
            "credit-challenger" if d["variant"] == "challenger" else "credit-scorecard"
        )
        assert d["model"] == expected
        assert d["decision"] == ("refer" if d["pd"] >= 0.5 else "approve")
    assert {c[1] for c in calls} == {"credit-scorecard", "credit-challenger"}
