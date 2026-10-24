"""The read-only guard in front of the Data API.

The assistant writes this SQL, so the guard is the one place that decides what a
model-generated statement is allowed to do. The warehouse grants are the real
backstop, but a guard that only inspects the first word is easy to walk past,
and these are the shapes that walk past it.
"""

import pytest

from helper import _rejects

WRITE_VERBS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "TRUNCATE",
    "ALTER",
    "GRANT",
    "REVOKE",
    "COPY",
    "UNLOAD",
    "CALL",
    "MERGE",
]


@pytest.mark.parametrize("verb", WRITE_VERBS)
def test_every_write_verb_is_rejected(verb):
    """Each verb named in the assistant's rules is refused."""
    assert _rejects(f"{verb} something")


@pytest.mark.parametrize("verb", WRITE_VERBS)
def test_case_and_leading_whitespace_do_not_help(verb):
    """Lower case and leading whitespace are the cheapest evasions to try."""
    assert _rejects(f"  \n\t{verb.lower()} something")


# ---------------------------------------------------------------------------
# The shapes that hide the verb from a first-word check
# ---------------------------------------------------------------------------
def test_a_block_comment_does_not_hide_the_verb():
    """A comment moves the verb off position zero, where a first-word check looks."""
    assert _rejects("/* harmless */ DELETE FROM analytics.applicants")


def test_a_line_comment_does_not_hide_the_verb():
    """The same evasion, written with a line comment instead."""
    assert _rejects("-- just looking\nDROP TABLE analytics.applicants")


def test_a_data_modifying_cte_is_rejected():
    """The statement opens with WITH, so the write sits well past position zero."""
    assert _rejects(
        "WITH gone AS (SELECT applicant_id FROM analytics.applicants) "
        "DELETE FROM analytics.applicants USING gone"
    )


def test_a_second_statement_is_rejected():
    """A write appended after a harmless read never reaches position zero."""
    assert _rejects("SELECT 1; DROP TABLE analytics.applicants")


def test_a_leading_bracket_does_not_hide_the_verb():
    """A bracket is one more character between the start and the verb."""
    assert _rejects("(DELETE FROM analytics.applicants)")


# ---------------------------------------------------------------------------
# Reads the analyst actually asks for must still run
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) AS n FROM analytics.applicant_credit_profile",
        "  select state, count(*) from analytics.applicant_credit_profile group by 1",
        "SELECT * FROM analytics.applicants LIMIT 10;",
        "-- how many per state?\nSELECT state, COUNT(*) FROM analytics.applicant_credit_profile GROUP BY 1",
        "WITH per_state AS (SELECT state, COUNT(*) AS n FROM analytics.applicant_credit_profile GROUP BY 1) SELECT * FROM per_state ORDER BY n DESC",
        "SELECT score::numeric / NULLIF(tradeline_count, 0) FROM analytics.applicant_credit_profile",
        "EXPLAIN SELECT 1",
    ],
)
def test_a_read_is_allowed(sql):
    """The queries the assistant is meant to run must still go through."""
    assert not _rejects(sql)


def test_a_trailing_semicolon_is_fine():
    """Ending one statement is not the same as starting a second."""
    assert not _rejects("SELECT 1;")


def test_a_write_verb_inside_a_string_literal_is_not_a_write():
    """A value that happens to read like SQL must not block a legitimate query."""
    assert not _rejects(
        "SELECT * FROM analytics.applicants WHERE full_name = 'DROP TABLE; x'"
    )


def test_a_write_verb_as_a_column_name_is_not_a_write():
    """A column called update_count must not be mistaken for an UPDATE."""
    assert not _rejects("SELECT update_count FROM analytics.applicants")
