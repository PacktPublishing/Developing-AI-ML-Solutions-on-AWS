# /// script
# dependencies = ["boto3", "ollama", "opensearch-py", "pdfplumber"]
# ///
"""Read a supporting document an underwriter adds to a case.

The bank statement is a table with known columns, so statements.py parses it and
the figures are arithmetic. Everything else an underwriter is sent is not: a tax
filing, an invoice, a supply contract, a page of management accounts. There is no
parser worth writing for that spread of layouts, so the model reads the file.

Amazon Bedrock takes the PDF itself, as a Converse document block, and Claude
Haiku 5.5 answers with the document's type, its figures, its dates and anything
that looks like a problem. The same call runs locally through the Ollama-backed
shim, where the difference is that the PDF is turned into text first and the
model reads the words rather than the file.

What comes back is treated as a reading, not as truth: it is stored beside the
document's own text, under the case, and the agent cites it as the document it
came from. The affordability arithmetic still comes from the statement.

Usage:
  PYTHONPATH=src uv run src/documents.py --case CASE-20260000 --file tax.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from models import DOC_MODEL, get_runtime
from statements import mask
from stores import get_store

# Converse takes a document up to 4.5 MB, and these are the formats it names.
MAX_BYTES = 4_500_000
FORMATS = {
    ".pdf": "pdf",
    ".csv": "csv",
    ".doc": "doc",
    ".docx": "docx",
    ".xls": "xls",
    ".xlsx": "xlsx",
    ".html": "html",
    ".txt": "txt",
    ".md": "md",
}

SYSTEM = (
    "You read one supporting document from a credit file and report what is in"
    " it. Report only what the document says. Leave a field empty rather than"
    " inferring it, and never estimate a figure that is not written down. Answer"
    " with one JSON object and no other text, in this shape:"
    ' {"doc_type": "", "issuer": "", "period": "", "currency": "",'
    ' "figures": [{"label": "", "amount": 0, "note": ""}],'
    ' "dates": [""], "concerns": [""], "summary": ""}.'
    " doc_type is what kind of document it is, for example tax filing, invoice,"
    " management accounts, supply contract. concerns is anything an underwriter"
    " should look at: a figure that contradicts another, a missing page, an"
    " unsigned document, a date out of range. summary is at most three"
    " sentences. Write plain prose with no em dashes, use commas instead."
)

ASK = (
    "Read this document and answer with the JSON object described. The"
    " underwriter is assessing a credit application and needs to know what the"
    " document is and what figures it carries."
)

# the model is asked for one object; a small local model wraps it in prose
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class Unsupported(ValueError):
    """Raised when a file cannot be sent to the document reader."""


@dataclass
class Reading:
    """What the model reported about one document, plus the document's text."""

    doc_name: str
    doc_type: str
    summary: str
    issuer: str = ""
    period: str = ""
    currency: str = ""
    figures: list[dict] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    # the model's own words when its answer was not the JSON that was asked for
    unparsed: str = ""
    pages: list[str] = field(default_factory=list)


def _block(path: Path) -> dict:
    """Return the Converse document block for one file, or refuse the file."""
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise Unsupported(
            f"{path.name}: {suffix or 'no extension'} is not a format the document"
            f" reader takes ({', '.join(sorted(FORMATS))})"
        )
    raw = path.read_bytes()
    if not raw:
        raise Unsupported(f"{path.name} is empty")
    if len(raw) > MAX_BYTES:
        raise Unsupported(
            f"{path.name} is {len(raw) / 1e6:.1f} MB, over the {MAX_BYTES / 1e6:.1f} MB"
            " a document block takes; split it or send the pages that matter"
        )
    return {
        "document": {
            "format": FORMATS[suffix],
            # the name is what the model calls the document in its answer
            "name": re.sub(r"[^A-Za-z0-9 \-()\[\]]", " ", path.stem)[:200]
            or "document",
            "source": {"bytes": raw},
        }
    }


def page_text(path: Path) -> list[str]:
    """Return the document's own text, one entry per page, masked.

    Stored beside the reading so an answer can quote the document rather than
    only the model's summary of it. A PDF of scanned images has no text layer
    and gives nothing here, which the caller reports rather than hides.
    """
    if path.suffix.lower() != ".pdf":
        return [mask(path.read_text(errors="replace"), "")]
    import pdfplumber  # lazy import: only this path needs it

    with pdfplumber.open(str(path)) as pdf:
        return [mask(page.extract_text() or "", "") for page in pdf.pages]


def read_document(path: str | Path, runtime, model: str = DOC_MODEL) -> Reading:
    """Send one document to the reader and return what it reported.

    Converse wants a text block beside the document block, so the question goes
    with the file in the same message.
    """
    path = Path(path)
    block = _block(path)
    resp = runtime.converse(
        modelId=model,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [block, {"text": ASK}]}],
        inferenceConfig={"maxTokens": 2000, "temperature": 0.0},
    )
    answer = resp["output"]["message"]["content"][0]["text"]
    name = block["document"]["name"]
    found = JSON_OBJECT.search(answer)
    if not found:
        return Reading(
            doc_name=name,
            doc_type="unread",
            summary="",
            unparsed=answer.strip(),
            pages=page_text(path),
        )
    try:
        fields = json.loads(found.group(0))
    except json.JSONDecodeError:
        return Reading(
            doc_name=name,
            doc_type="unread",
            summary="",
            unparsed=answer.strip(),
            pages=page_text(path),
        )
    figures = [f for f in fields.get("figures") or [] if isinstance(f, dict)]
    return Reading(
        doc_name=name,
        doc_type=str(fields.get("doc_type") or "unclassified"),
        summary=str(fields.get("summary") or ""),
        issuer=str(fields.get("issuer") or ""),
        period=str(fields.get("period") or ""),
        currency=str(fields.get("currency") or ""),
        figures=figures,
        dates=[str(d) for d in fields.get("dates") or []],
        concerns=[str(c) for c in fields.get("concerns") or [] if str(c).strip()],
        pages=page_text(path),
    )


def chunks_for(reading: Reading, case_id: str) -> list[str]:
    """Return the reading and the document's text as chunks to index.

    The first chunk is what the model made of the document, so a search for the
    document's subject finds it. The rest are its pages, so an answer can quote
    the document and the underwriter can check the quote.
    """
    figures = "; ".join(
        f"{f.get('label', '')} {f.get('amount', '')}".strip() for f in reading.figures
    )
    head = [
        f"Supporting document {reading.doc_name} for case {case_id}.",
        f"Type: {reading.doc_type}.",
    ]
    for label, value in (
        ("Issuer", reading.issuer),
        ("Period", reading.period),
        ("Currency", reading.currency),
        ("Figures", figures),
        ("Dates", ", ".join(reading.dates)),
        ("Concerns", "; ".join(reading.concerns)),
        ("Summary", reading.summary),
        ("Reader answered outside the requested format", reading.unparsed),
    ):
        if value:
            head.append(f"{label}: {value}")
    chunks = ["\n".join(head)]
    for number, text in enumerate(reading.pages, start=1):
        if text.strip():
            chunks.append(f"{reading.doc_name}, page {number}:\n{text}")
    return chunks


def submit(path: str | Path, case_id: str, store, runtime) -> Reading:
    """Read one supporting document and index it under the case."""
    if not case_id:
        raise Unsupported("a supporting document needs the case it belongs to")
    reading = read_document(path, runtime)
    store.index_submission(
        runtime,
        case_id,
        reading.doc_name,
        chunks_for(reading, case_id),
        meta={"doc_type": reading.doc_type},
        source="context",
        doc_name=reading.doc_name,
    )
    return reading


def main() -> None:
    """Read one document from the command line and index it under a case."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True, help="the case reference it belongs to")
    p.add_argument("--file", required=True, help="the document to read")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the reading without indexing it",
    )
    args = p.parse_args()

    runtime = get_runtime()
    if args.dry_run:
        reading = read_document(args.file, runtime)
    else:
        reading = submit(args.file, args.case, get_store(), runtime)
    where = "local" if os.environ.get("BEDROCK_LOCAL") == "1" else DOC_MODEL
    print(f"{reading.doc_name} read as {reading.doc_type} ({where})")
    if reading.summary:
        print(f"  {reading.summary}")
    for figure in reading.figures:
        print(f"  {figure.get('label', '')}: {figure.get('amount', '')}")
    for concern in reading.concerns:
        print(f"  concern: {concern}")
    if reading.unparsed:
        print(
            f"  the reader did not answer in the requested format: {reading.unparsed}"
        )
    if not any(page.strip() for page in reading.pages):
        print("  no text layer: nothing of the document itself could be indexed")


if __name__ == "__main__":
    main()
