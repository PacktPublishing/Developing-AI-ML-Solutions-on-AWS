"""Test batch response parsing, per-record errors and terminal job states."""

import json

import classify
import pytest
from classify import Run, collect, parse_label, parse_probs

LABELS = [
    "Bank Accounts",
    "Cards",
    "Customer Rewards",
    "Digital Banking",
    "Savings & Deposits",
]


def answer(text: str, stop_reason: str = "end_turn") -> dict:
    """Wrap text as the Converse response a batch record carries."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop_reason,
    }


class FakeS3:
    """The two S3 calls collect() makes, over an in-memory shard."""

    def __init__(self, lines: list[str]) -> None:
        """Hold the lines of the single .jsonl.out shard."""
        self.body = "\n".join(lines)

    def list_objects_v2(self, Bucket: str, Prefix: str) -> dict:
        """Return one record shard and the summary Bedrock writes beside it."""
        return {
            "Contents": [
                {"Key": f"{Prefix}manifest.json.out"},
                {"Key": f"{Prefix}manifest.jsonl.out"},
            ]
        }

    def get_object(self, Bucket: str, Key: str) -> dict:
        """Return the shard's body in the shape boto3 returns it."""

        class Body:
            """A stand-in for the StreamingBody boto3 hands back."""

            def __init__(self, text: str) -> None:
                self.text = text

            def read(self) -> bytes:
                """Return the bytes of the shard."""
                return self.text.encode("utf-8")

        return {"Body": Body(self.body)}


def gather(lines: list[str], distribution: bool = False):
    """Run collect() over an in-memory shard."""
    return collect(
        "bucket",
        "s3://bucket/output/run/",
        "arn:aws:bedrock:local:0:model-invocation-job/run",
        LABELS,
        FakeS3(lines),
        distribution,
    )


# ---------------------------------------------------------------------------
# One bad record is not the whole batch
# ---------------------------------------------------------------------------
def test_a_record_the_service_could_not_run_does_not_stop_collection():
    """Bedrock writes an error in place of modelOutput; the rest still counts."""
    preds, failures = gather(
        [
            json.dumps({"recordId": "1", "modelOutput": answer("Cards")}),
            json.dumps({"recordId": "2", "error": {"message": "throttled"}}),
            json.dumps({"recordId": "3", "modelOutput": answer("Digital Banking")}),
        ]
    )
    assert preds["1"]["pred"] == "Cards"
    assert preds["3"]["pred"] == "Digital Banking"
    assert "2" not in preds
    assert "throttled" in failures["2"]


def test_an_answer_that_is_not_a_converse_response_is_one_failed_record():
    """Record an invalid response as a per-record error."""
    preds, failures = gather(
        [
            json.dumps({"recordId": "1", "modelOutput": {"output": {}}}),
            json.dumps({"recordId": "2", "modelOutput": answer("Cards")}),
        ]
    )
    assert preds["2"]["pred"] == "Cards"
    assert "1" in failures


def test_an_unreadable_line_is_reported_rather_than_raised():
    """Report a malformed JSONL line without aborting collection."""
    preds, failures = gather(
        ["{not json", json.dumps({"recordId": "2", "modelOutput": answer("Cards")})]
    )
    assert preds["2"]["pred"] == "Cards"
    assert len(failures) == 1


# ---------------------------------------------------------------------------
# What counts as naming a category
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("Cards", "Cards"),
        ("cards", "Cards"),
        ("  Cards.  ", "Cards"),
        ("Category: Cards", "Cards"),
        ('"Cards"', "Cards"),
        ("```\nCards\n```", "Cards"),
        ("Savings and Deposits", "Savings & Deposits"),
        ("savings & deposits", "Savings & Deposits"),
        ("The conversation is about Cards", "Cards"),
    ],
)
def test_an_answer_that_names_one_category_is_taken(said, expected):
    """The cleanup is for formatting, and one mention is unambiguous."""
    label, why = parse_label(said, LABELS)
    assert (label, why) == (expected, "")


@pytest.mark.parametrize(
    "said",
    [
        "Not Cards. Digital Banking.",
        "Either Cards or Digital Banking",
        "Cards, but possibly Bank Accounts",
    ],
)
def test_an_answer_that_names_several_categories_is_rejected(said):
    """Reject responses mentioning multiple categories."""
    label, why = parse_label(said, LABELS)
    assert label == ""
    assert "names 2 categories" in why


@pytest.mark.parametrize("said", ["", "I am not sure", "Mortgages"])
def test_an_answer_that_names_no_known_category_is_rejected(said):
    """Reject responses without a recognized category."""
    label, why = parse_label(said, LABELS)
    assert label == ""
    assert why


def test_a_rejected_answer_becomes_a_failure_not_a_prediction():
    """Store rejected responses as failures without predictions."""
    preds, failures = gather(
        [
            json.dumps(
                {"recordId": "1", "modelOutput": answer("Not Cards. Digital Banking.")}
            )
        ]
    )
    assert preds == {}
    assert "1" in failures


# ---------------------------------------------------------------------------
# Probabilities are arithmetic, so they are validated
# ---------------------------------------------------------------------------
def test_a_valid_distribution_is_normalized_over_the_label_set():
    """The model is asked for about one; the vector is made to sum to one."""
    probs, why = parse_probs(json.dumps({"Cards": 0.6, "Digital Banking": 0.3}), LABELS)
    assert why == ""
    assert probs["Cards"] == pytest.approx(0.6 / 0.9)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert set(probs) == set(LABELS)


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ('{"Cards": "not sure"}', "not a number"),
        ('{"Cards": null}', "not a number"),
        ('{"Cards": true}', "not a number"),
        ('{"Cards": 1e309}', "not finite"),
        ('{"Cards": -0.5}', "outside"),
        ('{"Cards": 1.5}', "outside"),
        ('{"Cards": 0.2, "Digital Banking": 0.1}', "total"),
        ("{not json}", "does not parse"),
        ("no object here", "no JSON object"),
        ('{"Mortgages": 1.0}', "names none"),
        ("[0.5, 0.5]", "no JSON object"),
    ],
)
def test_a_distribution_that_is_not_one_is_refused_with_a_reason(raw, fragment):
    """Reject invalid probability distributions with an error reason."""
    probs, why = parse_probs(raw, LABELS)
    assert probs == {}
    assert fragment in why


def test_a_refused_distribution_is_a_failed_record():
    """Collection carries on, and the record is counted as failed."""
    preds, failures = gather(
        [
            json.dumps({"recordId": "1", "modelOutput": answer('{"Cards": "nope"}')}),
            json.dumps(
                {
                    "recordId": "2",
                    "modelOutput": answer('{"Cards": 0.9, "Digital Banking": 0.1}'),
                }
            ),
        ],
        distribution=True,
    )
    assert "1" in failures
    assert preds["2"]["pred"] == "Cards"


# ---------------------------------------------------------------------------
# The job's own ending
# ---------------------------------------------------------------------------
def test_the_terminal_states_are_the_ones_aws_defines():
    """Include Expired and PartiallyCompleted among terminal statuses."""
    assert set(classify.TERMINAL) == {
        "Completed",
        "PartiallyCompleted",
        "Failed",
        "Stopped",
        "Expired",
    }
    assert set(classify.USABLE) == {"Completed", "PartiallyCompleted"}


@pytest.mark.parametrize("status", ["Failed", "Stopped", "Expired"])
def test_a_job_that_did_not_finish_is_not_usable(status):
    """Reject failed, stopped and expired runs as unusable."""
    assert Run(status=status, message="").usable is False


@pytest.mark.parametrize("status", ["Completed", "PartiallyCompleted"])
def test_a_job_with_output_is_usable(status):
    """Accept completed and partially completed runs for collection."""
    assert Run(status=status, message="").usable is True


def test_every_run_gets_its_own_identifier():
    """Generate distinct identifiers for separate runs."""
    first, second = classify.new_run_id(), classify.new_run_id()
    assert first != second
    # a Bedrock job name takes letters, digits and hyphens
    assert all(part.isalnum() for part in f"classify-{first}".split("-"))


def test_an_answer_cut_off_at_the_token_limit_says_so():
    """Report token-limit truncation when the response stop reason is max_tokens."""
    preds, failures = gather(
        [
            json.dumps(
                {
                    "recordId": "1",
                    "modelOutput": answer("Let us think about this", "max_tokens"),
                }
            )
        ],
        distribution=True,
    )
    assert preds == {}
    assert "token limit" in failures["1"]


def test_a_malformed_answer_that_finished_is_not_blamed_on_the_limit():
    """The limit is only named when the service says that is what stopped the answer."""
    preds, failures = gather(
        [json.dumps({"recordId": "1", "modelOutput": answer("no idea sorry")})],
        distribution=True,
    )
    assert preds == {}
    assert "token limit" not in failures["1"]
