"""pgvector store: the local vector store, Amazon Aurora PostgreSQL on AWS."""

import os

import psycopg2

from models import embed


def connect():
    """Connect to the vector store from the standard PG environment variables."""
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5545"),
        user=os.environ.get("PGUSER", "underwriter"),
        password=os.environ.get("PGPASSWORD", "underwriter"),
        dbname=os.environ.get("PGDATABASE", "underwriting"),
    )


class PgVectorStore:
    """memo_chunks in pgvector: cosine k-NN over the embedded memo corpus."""

    def __init__(self) -> None:
        """Open an autocommit connection to the store."""
        self.conn = connect()
        self.conn.autocommit = True

    def reset(self, dim: int) -> None:
        """Create the extension and an empty memo_chunks table of width `dim`."""
        cur = self.conn.cursor()
        cur.execute("create extension if not exists vector")
        cur.execute(
            f"create table if not exists memo_chunks ("
            f" id bigserial primary key, loan_id bigint not null, borrower text not null,"
            f" chunk_index int not null, content text not null, embedding vector({dim}))"
        )
        cur.execute("truncate memo_chunks restart identity")
        cur.close()

    def add(
        self,
        loan_id: int,
        borrower: str,
        chunk_index: int,
        content: str,
        vector: list[float],
    ) -> None:
        """Insert one embedded chunk."""
        literal = "[" + ",".join(str(x) for x in vector) + "]"
        cur = self.conn.cursor()
        cur.execute(
            "insert into memo_chunks (loan_id, borrower, chunk_index, content, embedding)"
            " values (%s, %s, %s, %s, %s::vector)",
            (loan_id, borrower, chunk_index, content, literal),
        )
        cur.close()

    def finalize(self) -> None:
        """Build the HNSW index and close the connection."""
        cur = self.conn.cursor()
        cur.execute(
            "create index if not exists memo_chunks_embedding_idx"
            " on memo_chunks using hnsw (embedding vector_cosine_ops)"
        )
        cur.close()
        self.conn.close()

    def search(
        self, runtime, query: str, k: int = 5
    ) -> list[tuple[int, str, str, float]]:
        """Return the k nearest chunks as (loan_id, borrower, content, similarity)."""
        vector = embed(runtime, [query])[0]
        literal = "[" + ",".join(str(x) for x in vector) + "]"
        cur = self.conn.cursor()
        cur.execute(
            "select loan_id, borrower, content, 1 - (embedding <=> %s::vector) as similarity"
            " from memo_chunks order by embedding <=> %s::vector limit %s",
            (literal, literal, k),
        )
        rows = cur.fetchall()
        cur.close()
        return rows
