# /// script
# dependencies = ["boto3", "ollama", "opensearch-py"]
# ///
"""Embed the synthetic memo corpus into the vector store, one chunk per row.

Reads the memos written by gen_memos.py and the credit policy under data/policy,
splits each into chunks, embeds them through the model seam, and loads them into
the OpenSearch index for retrieval with citations back to the source loan or
policy. Each memo carries its loan-book fields so a search can filter on them.

Usage:
  make seed                 # embed the corpus into OpenSearch
"""

import argparse
from pathlib import Path

from models import EMBED_DIM, embed, get_runtime
from stores import get_store

CHUNK_SIZE = 800


def _header(path: Path) -> tuple[dict[str, str], str]:
    """Return a memo file's header fields and its body."""
    text = path.read_text(encoding="utf-8")
    header, _, body = text.partition("-" * 72)
    fields = {}
    for line in header.splitlines():
        if line.startswith("# ") and ":" in line:
            key, _, value = line[2:].partition(":")
            fields[key.strip()] = value.strip()
    return fields, body.strip()


def parse_memo(path: Path) -> tuple[int, str, str]:
    """Return (loan_id, borrower, body) from a memo file."""
    fields, body = _header(path)
    return int(fields["loan_id"]), fields["borrower"], body


def memo_meta(path: Path) -> dict:
    """Return the loan-book fields a search filters on, leaving out blank ones."""
    fields, _ = _header(path)
    meta: dict = {"decided": fields["type"].split("uploaded:")[-1].strip()[:10]}
    if fields.get("sector"):
        meta["sector"] = fields["sector"]
    for key in ("exposure_usd", "tenor_months"):
        if int(fields.get(key) or 0):
            meta[key] = int(fields[key])
    return meta


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


def seed(memo_dir: Path, policy_dir: Path | None = None) -> int:
    """Embed every memo and policy document into the store; return the chunk count."""
    runtime = get_runtime()
    store = get_store()
    store.reset(EMBED_DIM)

    total = 0
    for path in sorted(memo_dir.glob("*.txt")):
        loan_id, borrower, body = parse_memo(path)
        meta = memo_meta(path)
        chunks = chunk_text(body)
        vectors = embed(runtime, chunks)
        for i, (content, vector) in enumerate(zip(chunks, vectors)):
            store.add(loan_id, borrower, i, content, vector, meta=meta)
            total += 1

    # policy has no loan id, so it is stored under its policy id and cited by it
    for path in sorted(policy_dir.glob("*.md")) if policy_dir else []:
        chunks = chunk_text(path.read_text(encoding="utf-8"))
        vectors = embed(runtime, chunks)
        for i, (content, vector) in enumerate(zip(chunks, vectors)):
            store.add(0, path.stem, i, content, vector, source="policy")
            total += 1

    store.finalize()
    return total


def main() -> None:
    """Embed the generated memo corpus into the vector store."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memos", type=Path, default=Path("data/generated/memos"))
    p.add_argument("--policy", type=Path, default=Path("data/policy"))
    a = p.parse_args()
    total = seed(a.memos, a.policy)
    print(f"Seeded {total} chunks into memo_chunks from {a.memos} and {a.policy}")


if __name__ == "__main__":
    main()
