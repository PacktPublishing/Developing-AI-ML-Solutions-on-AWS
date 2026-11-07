"""The decision-model router: the question it asks and what it does with the answer.

No model is loaded here. A decision model's answer is already structured, so the
only things left to get wrong are the question put to it and the policy applied
to its confidence, and both are worth pinning down.
"""

import categories
import pytest
from decide import question_for, record_for, route_one

LABELS = [
    "Bank Accounts",
    "Cards",
    "Digital Banking",
    "Savings & Deposits",
]


class FakeDecider:
    """A decision model that answers with whatever it was handed."""

    device = "fake"

    def __init__(self, team: str, probabilities: dict, confidence: float):
        """Hold the answer to return and record the question that was asked."""
        self.answer = {
            "type": "choice",
            "choice": team,
            "probabilities": probabilities,
            "confidence": confidence,
        }
        self.asked: list[tuple[str, dict]] = []

    def decide(self, state: str, questions: dict) -> dict:
        """Return the canned answer and remember the call."""
        self.asked.append((state, questions))
        return {"model": "fake", "answers": {"team": self.answer}}


def test_bare_names_ask_about_every_team_without_describing_any():
    """Variant a names the teams and says nothing about where their work begins."""
    question = question_for(LABELS, described=False)["team"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == LABELS
    assert set(question["criteria"].values()) == {None}


def test_defined_borders_carry_a_definition_for_every_team():
    """Variant b describes each option, which is where a decision model reads borders."""
    question = question_for(LABELS, described=True)["team"]
    assert list(question["criteria"]) == LABELS
    assert all(question["criteria"][label] for label in LABELS)
    assert "fixed/term deposits" in question["criteria"]["Savings & Deposits"]


def test_the_llm_and_the_decision_model_are_briefed_from_one_text():
    """A comparison of two routers has to be a comparison of the routers.

    The definitions live in categories.py precisely so that neither router can
    quietly get a better brief than the other.
    """
    from classify import INSTRUCTION_B

    for team, border in categories.BORDERS.items():
        assert f"- {team}: {border}." in INSTRUCTION_B
        assert categories.as_criteria([team], described=True)[team] == border


def test_a_team_the_dataset_does_not_use_is_not_offered():
    """The options are the dataset's labels, not every team ever defined."""
    criteria = categories.as_criteria(["Cards"], described=True)
    assert list(criteria) == ["Cards"]


def test_the_answer_is_read_without_parsing_any_text():
    """The chosen team, the distribution and the confidence come back as fields."""
    decider = FakeDecider("Cards", {"Cards": 0.8, "Bank Accounts": 0.2}, 0.73)
    team, probs, confidence = route_one(decider, "my card was declined", {"team": {}})
    assert (team, confidence) == ("Cards", 0.73)
    assert probs == {"Cards": 0.8, "Bank Accounts": 0.2}
    assert decider.asked[0][0] == "my card was declined"


def test_a_confident_answer_is_routed():
    """Above the threshold the team stands and the distribution is kept for scoring."""
    record = record_for(
        {"id": "1", "label": "Cards"}, "Cards", {"Cards": 0.9}, 0.9, 0.6, "run"
    )
    assert record["pred"] == "Cards"
    assert record["probs"] == {"Cards": 0.9}
    assert "failed" not in record


def test_a_close_call_is_left_for_a_person_rather_than_routed():
    """Below the threshold nothing is routed, and the reason names the threshold.

    An empty pred is what the scorer counts as uncovered, so a held conversation
    lowers coverage instead of being scored as a confident wrong answer.
    """
    record = record_for(
        {"id": "1", "label": "Cards"}, "Cards", {"Cards": 0.3}, 0.3, 0.6, "run"
    )
    assert record["pred"] == ""
    assert record["probs"] is None
    assert "0.300" in record["failed"]
    assert "0.60" in record["failed"]


def test_a_threshold_of_zero_routes_everything():
    """The default policy routes every conversation, however close the call."""
    record = record_for(
        {"id": "1", "label": "Cards"}, "Cards", {"Cards": 0.11}, 0.0, 0.0, "run"
    )
    assert record["pred"] == "Cards"


@pytest.mark.parametrize("described", [True, False])
def test_either_variant_asks_exactly_one_question(described):
    """One conversation, one question, so one forward pass answers it."""
    assert list(question_for(LABELS, described=described)) == ["team"]
