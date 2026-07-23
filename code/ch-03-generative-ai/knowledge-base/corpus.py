# /// script
# dependencies = ["boto3", "psycopg2-binary", "ollama"]
# ///
"""Build the knowledge base: generate it and embed it into the vector store.

The lender has moved into corporate lending to the residential and commercial
solar sector: installers, developers, and the project companies that own the
assets. These are young businesses in a new market, so there is no external
credit score to lean on. An underwriter reads the documents, compares the deal
against the sector, and assigns an internal risk profile by hand. The knowledge
base is what supports that judgment: short sector profiles and the lender's own
credit policy. This script generates those documents and embeds them straight
into pgvector.

All content is synthetic. It reflects the shape of solar-sector corporate
lending without reproducing any lender's actual policy or any borrower's data.

Usage:
  uv run knowledge-base/corpus.py
"""

import json
import pathlib

from corpus_data import AUTHORITY_TIERS, chunk_text, documents
from models import EMBED_DIM, embed, get_runtime
from stores import connect


def seed() -> None:
    """Generate the knowledge base, embed it into pgvector, and write the matrix."""
    data_dir = pathlib.Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "authority_limits.json").write_text(
        json.dumps(AUTHORITY_TIERS, indent=2)
    )

    runtime = get_runtime()
    conn = connect()
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("create extension if not exists vector")
    cur.execute(
        f"create table if not exists kb_chunks ("
        f" id bigserial primary key, doc_id text not null, chunk_index int not null,"
        f" content text not null, embedding vector({EMBED_DIM}))"
    )
    cur.execute("truncate kb_chunks restart identity")

    total = 0
    for doc_id, text in documents():
        chunks = chunk_text(text)
        vectors = embed(runtime, chunks)
        for i, (content, vector) in enumerate(zip(chunks, vectors)):
            literal = "[" + ",".join(str(x) for x in vector) + "]"
            cur.execute(
                "insert into kb_chunks (doc_id, chunk_index, content, embedding)"
                " values (%s, %s, %s, %s::vector)",
                (doc_id, i, content, literal),
            )
            total += 1

    cur.execute(
        "create index if not exists kb_chunks_embedding_idx"
        " on kb_chunks using hnsw (embedding vector_cosine_ops)"
    )
    cur.close()
    conn.close()
    print(f"Seeded {total} chunks into kb_chunks")
    print("Wrote the authority-limits matrix to data/authority_limits.json")


def main() -> None:
    """Generate the knowledge base and embed it into the vector store."""
    seed()


if __name__ == "__main__":
    main()
