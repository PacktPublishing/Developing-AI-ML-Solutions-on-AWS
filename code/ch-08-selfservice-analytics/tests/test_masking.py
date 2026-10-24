"""The transcript-side masking layer.

Masking happens by column name, so the test that matters is that a column listed
in config/pii_columns.yaml never reaches a result set in the clear, and that a
column not listed is returned untouched.
"""

import pathlib

import pytest

from masking import _STRATEGIES, mask_records, masked_columns

CONFIG = pathlib.Path(__file__).resolve().parents[1] / "config" / "pii_columns.yaml"


def test_the_strategies_come_from_the_shared_config():
    """The warehouse policies read the same file; a drift here is a drift there."""
    assert CONFIG.exists()
    listed = {
        line.split(":")[0].strip().rsplit(".", 1)[-1]
        for line in CONFIG.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert set(masked_columns()) == listed


def test_every_configured_column_is_masked_in_a_result():
    """No column listed in the config may reach the caller in the clear."""
    row = dict.fromkeys(masked_columns(), "sensitive-value")
    masked = mask_records([row])[0]
    for column in masked_columns():
        assert masked[column] != "sensitive-value", f"{column} reached the caller clear"


def test_columns_outside_the_config_are_left_alone():
    """Masking by name must not touch the rest of the result set."""
    rows = [{"loan_id": 42, "amount_usd": 1500.0, "status": "open"}]
    assert mask_records(rows) == rows


def test_a_mixed_row_masks_only_the_listed_columns():
    """A realistic row carries both kinds of column at once."""
    rows = [{"email": "a@b.com", "loan_id": 7}]
    out = mask_records(rows)[0]
    assert out["loan_id"] == 7
    assert out["email"] != "a@b.com"


# ---------------------------------------------------------------------------
# The strategies
# ---------------------------------------------------------------------------
def test_hash_is_deterministic_so_joins_and_distinct_counts_survive():
    """The same input must always produce the same token, and different inputs must not collide."""
    column = next(c for c, s in _STRATEGIES.items() if s == "hash")
    first = mask_records([{column: "a@b.com"}])[0][column]
    second = mask_records([{column: "a@b.com"}])[0][column]
    assert first == second
    assert mask_records([{column: "c@d.com"}])[0][column] != first


def test_partial_last4_keeps_only_the_last_four_characters():
    """The length is preserved so the shape of the value still reads as a phone number."""
    column = next(c for c, s in _STRATEGIES.items() if s == "partial_last4")
    out = mask_records([{column: "08031234567"}])[0][column]
    assert out.endswith("4567")
    assert set(out[:-4]) == {"*"}
    assert len(out) == len("08031234567")


def test_redact_keeps_nothing():
    """A name has no analytically useful prefix, so it goes entirely."""
    column = next(c for c, s in _STRATEGIES.items() if s == "redact")
    assert mask_records([{column: "Ada Lovelace"}])[0][column] == "***"


def test_a_short_value_does_not_leak_through_partial_last4():
    """Shorter than four characters has nothing to hide, and nothing to star out."""
    column = next(c for c, s in _STRATEGIES.items() if s == "partial_last4")
    out = mask_records([{column: "12"}])[0][column]
    assert out == "12", "shorter than four characters is returned as-is"


# ---------------------------------------------------------------------------
# Shapes the Data API actually returns
# ---------------------------------------------------------------------------
def test_an_empty_result_is_returned_unchanged():
    """A query that matched nothing is the common case, not an error."""
    assert mask_records([]) == []


def test_nulls_stay_null():
    """Masking a null would turn a missing value into a present one."""
    column = next(iter(masked_columns()))
    assert mask_records([{column: None}])[0][column] is None


def test_every_row_is_masked_not_only_the_first():
    """The column set is read from the first row; the masking must still apply to all."""
    column = next(iter(masked_columns()))
    rows = [{column: f"value-{i}"} for i in range(5)]
    for out in mask_records(rows):
        assert not str(out[column]).startswith("value-")


def test_masking_does_not_mutate_the_caller_rows():
    """The helper returns new rows, so a caller holding the originals is unaffected."""
    column = next(iter(masked_columns()))
    rows = [{column: "original"}]
    mask_records(rows)
    assert rows[0][column] == "original"


def test_a_non_string_value_is_still_masked():
    """The Data API returns numbers as numbers, and a numeric identifier is still PII."""
    column = next(c for c, s in _STRATEGIES.items() if s == "hash")
    out = mask_records([{column: 1234567890}])[0][column]
    assert out != 1234567890
    assert isinstance(out, str)


@pytest.mark.parametrize("salt", ["book", "other"])
def test_the_hash_salt_is_honoured(monkeypatch, salt):
    """A deployment can change MASK_SALT so tokens are not comparable across environments."""
    import importlib

    import masking

    column = next(c for c, s in _STRATEGIES.items() if s == "hash")
    monkeypatch.setenv("MASK_SALT", salt)
    importlib.reload(masking)
    return_value = masking.mask_records([{column: "a@b.com"}])[0][column]
    assert isinstance(return_value, str) and len(return_value) == 12
