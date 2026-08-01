"""The vector store seam: pgvector locally, RDS for PostgreSQL on AWS.

The embeddings live in Postgres with pgvector, a container locally and Amazon
RDS for PostgreSQL on AWS. The retrieval code is the same in either world; only
the host moves, through the standard PG environment variables.
"""

import os

import psycopg2

from models import embed


def connect():
    """Connect to the vector store from the standard PG environment variables."""
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5544"),
        user=os.environ.get("PGUSER", "underwriter"),
        password=os.environ.get("PGPASSWORD", "underwriter"),
        dbname=os.environ.get("PGDATABASE", "underwriting"),
    )


def search(runtime, query: str, k: int = 5) -> list[tuple[str, str, float]]:
    """Return the k nearest knowledge-base chunks as (doc_id, content, similarity)."""
    query_vector = embed(runtime, [query])[0]
    literal = "[" + ",".join(str(x) for x in query_vector) + "]"
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "select doc_id, content, 1 - (embedding <=> %s::vector) as similarity"
        " from kb_chunks order by embedding <=> %s::vector limit %s",
        (literal, literal, k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
