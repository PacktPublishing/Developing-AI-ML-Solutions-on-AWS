"""The local KYC vector store: connect and create the schema, index-first."""

import os

import psycopg

DSN = os.environ.get("KYC_DSN", "postgresql://kyc:kyc@localhost:5507/kyc")


def connect():
    """Open a connection to the local pgvector store."""
    return psycopg.connect(DSN, connect_timeout=10)


def ensure_schema(conn) -> None:
    """Create the faces table and its HNSW index up front (index-first)."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id        BIGSERIAL PRIMARY KEY,
                subject   TEXT NOT NULL,
                s3_key    TEXT UNIQUE NOT NULL,
                embedding VECTOR(512)
            )
        """)
        # the HNSW graph exists before any rows, so each enrol write maintains
        # the index incrementally rather than paying for a rebuild later.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS faces_hnsw ON faces "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    conn.commit()
