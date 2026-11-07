"""The document reader: what is sent, what is refused, what comes back.

The reading itself is a model's work and is not asserted here. What is asserted
is the contract around it: the file goes to Bedrock as a document block, a file
Converse will not take is refused before the call, an answer that is not the
JSON that was asked for is kept as the model's words rather than invented into
fields, and the document's own text is indexed beside the reading so an answer
can be checked against the page.
"""

import json

import pytest

pytest.importorskip("pdfplumber")
pytest.importorskip("reportlab")

import documents
from documents import MAX_BYTES, Reading, Unsupported, chunks_for, read_document

ANSWER = {
    "doc_type": "tax filing",
    "issuer": "Federal Inland Revenue",
    "period": "2025",
    "currency": "USD",
    "figures": [
        {"label": "turnover", "amount": 18_400_000, "note": "declared"},
        "not a figure",
    ],
    "dates": ["2026-03-31"],
    "concerns": ["the second page is missing", ""],
    "summary": "A 2025 tax filing declaring turnover of 18.4 million.",
}


class FakeRuntime:
    """A bedrock-runtime stand-in that records the call and answers with text."""

    def __init__(self, text: str) -> None:
        """Hold the answer to give and somewhere to keep the request."""
        self.text = text
        self.call: dict = {}

    def converse(self, modelId, messages, system=None, inferenceConfig=None) -> dict:
        """Record the request and return it in the Converse response shape."""
        self.call = {
            "modelId": modelId,
            "messages": messages,
            "system": system,
            "inferenceConfig": inferenceConfig,
        }
        return {
            "output": {
                "message": {"role": "assistant", "content": [{"text": self.text}]}
            }
        }


@pytest.fixture
def filing(tmp_path):
    """Write a one page PDF with a line of text in it."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = tmp_path / "tax filing 2025.pdf"
    page = canvas.Canvas(str(path), pagesize=A4)
    page.drawString(80, 760, "Turnover for 2025: 18,400,000")
    page.save()
    return path


# ---------------------------------------------------------------------------
# What goes to the model
# ---------------------------------------------------------------------------
def test_the_pdf_itself_is_sent_as_a_document_block(filing):
    """Bedrock reads the file, so the bytes travel, not a transcription."""
    runtime = FakeRuntime(json.dumps(ANSWER))
    read_document(filing, runtime, model="test-model")
    content = runtime.call["messages"][0]["content"]
    block = content[0]["document"]
    assert block["format"] == "pdf"
    assert block["source"]["bytes"] == filing.read_bytes()
    # Converse requires a text block beside the document block
    assert any("text" in part for part in content)
    assert runtime.call["messages"][0]["role"] == "user"


def test_the_document_name_survives_but_loses_what_a_name_may_not_hold(filing):
    """The model refers to the document by name, so the name has to be clean."""
    runtime = FakeRuntime(json.dumps(ANSWER))
    reading = read_document(filing, runtime)
    assert reading.doc_name == "tax filing 2025"


# ---------------------------------------------------------------------------
# What is refused, before the call
# ---------------------------------------------------------------------------
def test_a_format_converse_does_not_take_is_refused(tmp_path):
    """A refusal here costs nothing; a refusal from the service costs a call."""
    path = tmp_path / "scan.tiff"
    path.write_bytes(b"II*\x00")
    with pytest.raises(Unsupported, match="not a format"):
        read_document(path, FakeRuntime("{}"))


def test_an_empty_file_is_refused(tmp_path):
    """An empty upload is a failed upload, not a document with nothing in it."""
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    with pytest.raises(Unsupported, match="empty"):
        read_document(path, FakeRuntime("{}"))


def test_a_file_over_the_document_block_limit_is_refused(tmp_path):
    """Converse takes 4.5 MB, and the caller is told which it was."""
    path = tmp_path / "big.pdf"
    path.write_bytes(b"0" * (MAX_BYTES + 1))
    with pytest.raises(Unsupported, match="MB"):
        read_document(path, FakeRuntime("{}"))


def test_a_supporting_document_needs_a_case(filing):
    """Filed under no case, a document answers for every case."""
    with pytest.raises(Unsupported, match="case"):
        documents.submit(filing, "", None, FakeRuntime(json.dumps(ANSWER)))


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------
def test_the_reading_carries_the_fields_the_model_filled_in(filing):
    """The fields are read straight through; nothing is derived from them."""
    reading = read_document(filing, FakeRuntime(json.dumps(ANSWER)))
    assert reading.doc_type == "tax filing"
    assert reading.issuer == "Federal Inland Revenue"
    assert reading.figures == [ANSWER["figures"][0]]
    assert reading.concerns == ["the second page is missing"]
    assert reading.unparsed == ""


def test_json_wrapped_in_prose_is_still_read(filing):
    """A small local model explains itself first; the object is still in there."""
    runtime = FakeRuntime(
        f"Here is the answer:\n{json.dumps(ANSWER)}\nHope that helps."
    )
    assert read_document(filing, runtime).doc_type == "tax filing"


@pytest.mark.parametrize("answer", ["I could not read this document.", "{not json}"])
def test_an_answer_that_is_not_the_requested_json_is_kept_as_words(filing, answer):
    """Inventing fields out of a refusal is worse than reporting the refusal."""
    reading = read_document(filing, FakeRuntime(answer))
    assert reading.doc_type == "unread"
    assert reading.unparsed == answer
    assert reading.figures == []


def test_the_documents_own_text_is_indexed_beside_the_reading(filing):
    """An answer the underwriter cannot check against the page is not evidence."""
    reading = read_document(filing, FakeRuntime(json.dumps(ANSWER)))
    assert any("18,400,000" in page for page in reading.pages)
    chunks = chunks_for(reading, "CASE-1")
    assert chunks[0].startswith("Supporting document tax filing 2025 for case CASE-1")
    assert "tax filing" in chunks[0]
    assert any("page 1" in chunk for chunk in chunks[1:])


def test_a_page_with_no_text_layer_adds_no_chunk():
    """A scan has nothing to index, and an empty chunk is worse than none."""
    reading = Reading(doc_name="scan", doc_type="unread", summary="", pages=["", "   "])
    assert len(chunks_for(reading, "CASE-1")) == 1


def test_an_email_in_a_supporting_document_is_masked(tmp_path):
    """The chunks are embedded and stored, here as much as for the statement."""
    path = tmp_path / "invoice.txt"
    path.write_text(
        "Invoice 338. Contact billing@supplier.example for 4929123456789012."
    )
    reading = read_document(path, FakeRuntime(json.dumps(ANSWER)))
    page = reading.pages[0]
    assert "billing@supplier.example" not in page
    assert "4929123456789012" not in page
    assert "[EMAIL]" in page and "[NUMBER]" in page


# ---------------------------------------------------------------------------
# The local path reads the words instead of the file
# ---------------------------------------------------------------------------
def test_the_local_shim_turns_a_document_block_into_text(filing):
    """Ollama takes text, so this is where the local path has to differ."""
    from models import _as_text

    block = {
        "document": {
            "format": "pdf",
            "name": "tax filing 2025",
            "source": {"bytes": filing.read_bytes()},
        }
    }
    flattened = _as_text([block, {"text": "read this"}])
    assert "18,400,000" in flattened
    assert "tax filing 2025" in flattened
    assert "read this" in flattened
