# /// script
# dependencies = ["boto3", "psycopg2-binary", "ollama"]
# ///
"""Embed the synthetic memo corpus into the vector store, one chunk per row.

Reads the memos written by gen_memos.py, splits each into chunks, embeds them
through the model seam, and loads them into pgvector for retrieval with
citations back to the source loan.

Usage:
  make seed    # or: PYTHONPATH=src uv run etl/embed_memos.py --memos data/generated/memos
"""

import argparse
from pathlib import Path

from models import EMBED_DIM, embed, get_runtime
from stores import connect

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
    """Embed every memo under `memo_dir` into memo_chunks and return the count."""
    runtime = get_runtime()
    conn = connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("create extension if not exists vector")
    cur.execute(
        f"create table if not exists memo_chunks ("
        f" id bigserial primary key, loan_id bigint not null, borrower text not null,"
        f" chunk_index int not null, content text not null, embedding vector({EMBED_DIM}))"
    )
    cur.execute("truncate memo_chunks restart identity")

    total = 0
    for path in sorted(memo_dir.glob("*.txt")):
        loan_id, borrower, body = parse_memo(path)
        chunks = chunk_text(body)
        vectors = embed(runtime, chunks)
        for i, (content, vector) in enumerate(zip(chunks, vectors)):
            literal = "[" + ",".join(str(x) for x in vector) + "]"
            cur.execute(
                "insert into memo_chunks (loan_id, borrower, chunk_index, content, embedding)"
                " values (%s, %s, %s, %s, %s::vector)",
                (loan_id, borrower, i, content, literal),
            )
            total += 1

    cur.execute(
        "create index if not exists memo_chunks_embedding_idx"
        " on memo_chunks using hnsw (embedding vector_cosine_ops)"
    )
    cur.close()
    conn.close()
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
