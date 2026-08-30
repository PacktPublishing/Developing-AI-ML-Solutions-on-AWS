# /// script
# dependencies = ["boto3", "ollama", "opensearch-py"]
# ///
"""Embed the synthetic memo corpus into the vector store, one chunk per row.

Reads the memos written by gen_memos.py, splits each into chunks, embeds them
through the model seam, and loads them into the OpenSearch index for retrieval
with citations back to the source loan.

Usage:
  make seed                 # embed the corpus into OpenSearch
"""

import argparse
from pathlib import Path

from models import EMBED_DIM, embed, get_runtime
from stores import get_store

CHUNK_SIZE = 800


def parse_memo(path: Path) -> tuple[int, str, str]:
    """Return (loan_id, borrower, body) from a memo file's 4-line header."""
    text = path.read_text(encoding="utf-8")
    header, _, body = text.partition("-" * 72)
    fields = {}
    for line in header.splitlines():
        if line.startswith("# ") and ":" in line:
            key, _, value = line[2:].partition(":")
            fields[key.strip()] = value.strip()
    return int(fields["loan_id"]), fields["borrower"], body.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks of about `size` characters on paragraph breaks."""
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def seed(memo_dir: Path) -> int:
    """Embed every memo under `memo_dir` into the selected store; return the count."""
    runtime = get_runtime()
    store = get_store()
    store.reset(EMBED_DIM)

    total = 0
    for path in sorted(memo_dir.glob("*.txt")):
        loan_id, borrower, body = parse_memo(path)
        chunks = chunk_text(body)
        vectors = embed(runtime, chunks)
        for i, (content, vector) in enumerate(zip(chunks, vectors)):
            store.add(loan_id, borrower, i, content, vector)
            total += 1

    store.finalize()
    return total


def main() -> None:
    """Embed the generated memo corpus into the vector store."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memos", type=Path, default=Path("data/generated/memos"))
    a = p.parse_args()
    total = seed(a.memos)
    print(f"Seeded {total} chunks into memo_chunks from {a.memos}")


if __name__ == "__main__":
    main()
