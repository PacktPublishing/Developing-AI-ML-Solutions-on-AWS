"""Test decision-model questions and confidence thresholds with a stub model."""

import categories
import pytest
from decide import question_for, record_for, route_one

META = {"run_id": "run", "model": "fake", "backend": "fake"}

LABELS = [
    "Bank Accounts",
    "Cards",
    "Digital Banking",
    "Savings & Deposits",
]


class FakeDecider:
    """A decision model that answers with whatever it was handed."""

    where = "fake"

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
    """Include each category name without definitions in variant a."""
    question = question_for(LABELS, described=False)["team"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == LABELS
    assert set(question["criteria"].values()) == {None}


def test_defined_borders_carry_a_definition_for_every_team():
    """Include each category definition in variant b."""
    question = question_for(LABELS, described=True)["team"]
    assert list(question["criteria"]) == LABELS
    assert all(question["criteria"][label] for label in LABELS)
    assert "fixed/term deposits" in question["criteria"]["Savings & Deposits"]


def test_the_llm_and_the_decision_model_are_briefed_from_one_text():
    """Use the same category definitions for both routers."""
    from classify import INSTRUCTION_B

    for team, border in categories.BORDERS.items():
        assert f"- {team}: {border}." in INSTRUCTION_B
        assert categories.as_criteria([team], described=True)[team] == border


def test_a_team_the_dataset_does_not_use_is_not_offered():
    """Offer only categories present in the dataset label set."""
    criteria = categories.as_criteria(["Cards"], described=True)
    assert list(criteria) == ["Cards"]


def test_the_answer_is_read_without_parsing_any_text():
    """The chosen team, the distribution and the confidence come back as fields."""
    decider = FakeDecider("Cards", {"Cards": 0.8, "Bank Accounts": 0.2}, 0.73)
    team, probs, confidence, model = route_one(
        decider, "my card was declined", {"team": {}}
    )
    assert (team, confidence) == ("Cards", 0.73)
    assert probs == {"Cards": 0.8, "Bank Accounts": 0.2}
    assert model == "fake"
    assert decider.asked[0][0] == "my card was declined"


def test_a_confident_answer_is_routed():
    """Above the threshold the team stands and the distribution is kept for scoring."""
    record = record_for(
        {"id": "1", "label": "Cards"}, "Cards", {"Cards": 0.9}, 0.9, 0.6, META
    )
    assert record["pred"] == "Cards"
    assert record["probs"] == {"Cards": 0.9}
    assert "failed" not in record


def test_a_close_call_is_left_for_a_person_rather_than_routed():
    """Clear predictions below the confidence threshold and record the reason."""
    record = record_for(
        {"id": "1", "label": "Cards"}, "Cards", {"Cards": 0.3}, 0.3, 0.6, META
    )
    assert record["pred"] == ""
    assert record["probs"] is None
    assert "0.300" in record["failed"]
    assert "0.60" in record["failed"]


def test_a_threshold_of_zero_routes_everything():
    """The default policy routes every conversation, however close the call."""
    record = record_for(
        {"id": "1", "label": "Cards"}, "Cards", {"Cards": 0.11}, 0.0, 0.0, META
    )
    assert record["pred"] == "Cards"


@pytest.mark.parametrize("described", [True, False])
def test_either_variant_asks_exactly_one_question(described):
    """Generate one choice question for either prompt variant."""
    assert list(question_for(LABELS, described=described)) == ["team"]


def test_the_record_says_which_model_answered():
    """Two backends write into the same file shape, so provenance travels with it.

    Comparing a Strands Decider run against a Kev run later depends on the rows
    saying which one produced them.
    """
    record = record_for(
        {"id": "1", "label": "Cards"}, "Cards", {"Cards": 0.9}, 0.9, 0.0, META
    )
    assert record["model"] == "fake"
    assert record["backend"] == "fake"
    assert record["run_id"] == "run"


@pytest.mark.parametrize("path", [None, "", "/invocations"])
def test_http_endpoint_path(monkeypatch, path):
    """Use the System One path unless an explicit endpoint path is supplied."""
    import importlib
    import decider

    with monkeypatch.context() as env:
        if path is None:
            env.delenv("DECIDER_PATH", raising=False)
        else:
            env.setenv("DECIDER_PATH", path)
        importlib.reload(decider)
        client = decider.HttpDecider("http://localhost:8019/")
        assert client._url == "http://localhost:8019" + (path or "/v1/systemone")
    importlib.reload(decider)


@pytest.mark.parametrize(
    "target,local_credentials",
    [("up", True), ("classify", True), ("serve-agent", False)],
)
def test_make_credentials_are_scoped_to_s3proxy(target, local_credentials):
    """Cloud callers must not inherit the S3Proxy placeholder credentials."""
    import os
    from pathlib import Path
    import subprocess

    chapter = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-s", "--eval", f"{target}: ; @env", target],
        cwd=chapter,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if local_credentials:
        assert values["AWS_ACCESS_KEY_ID"] == "local"
        assert values["AWS_SECRET_ACCESS_KEY"] == "localsecret"
    else:
        assert "AWS_ACCESS_KEY_ID" not in values
        assert "AWS_SECRET_ACCESS_KEY" not in values
