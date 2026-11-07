"""Test paragraph, line and character splitting against the chunk-size limit."""

from chunking import CHUNK_SIZE, chunk_text


def test_short_text_is_one_chunk():
    """Keep text shorter than the limit in one chunk."""
    assert chunk_text("a tax filing for 2026") == ["a tax filing for 2026"]


def test_paragraphs_are_preferred_boundaries():
    """Split at paragraph boundaries when they fit the limit."""
    first, second = "x" * 500, "y" * 500
    assert chunk_text(f"{first}\n\n{second}") == [first, second]


def test_blank_paragraphs_are_dropped():
    """Repeated blank lines in a PDF extraction do not become empty chunks."""
    assert chunk_text("one\n\n\n\n\n\ntwo") == ["one\n\ntwo"]


def test_a_page_of_rows_is_still_capped():
    """Split a page of transaction rows into chunks within the character limit."""
    page = "\n".join(
        f"2026-03-{day:02d} POS PURCHASE FUEL STATION 42.00" for day in range(1, 60)
    )
    assert len(page) > CHUNK_SIZE
    chunks = chunk_text(page)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_SIZE for c in chunks)


def test_no_chunk_ever_exceeds_the_cap():
    """Keep each tested chunk within the character limit."""
    for text in (
        "z" * 5000,
        "\n".join("row " * 40 for _ in range(40)),
        ("para " * 100 + "\n\n") * 10,
    ):
        assert all(len(c) <= CHUNK_SIZE for c in chunk_text(text))


def test_a_single_line_longer_than_the_cap_is_cut():
    """Split an oversized line at the character limit."""
    chunks = chunk_text("q" * (CHUNK_SIZE * 2 + 50))
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    assert "".join(chunks) == "q" * (CHUNK_SIZE * 2 + 50)


def test_nothing_is_lost_when_rows_are_split():
    """Preserve all rows when splitting text."""
    rows = [f"row {i} amount {i * 11}" for i in range(200)]
    rejoined = "\n".join(chunk_text("\n".join(rows)))
    for row in rows:
        assert row in rejoined
