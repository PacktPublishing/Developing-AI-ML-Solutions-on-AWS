"""The vector-store seam: one interface, three backends, selected by STORE.

STORE=pgvector (default) uses Postgres with pgvector; STORE=opensearch uses an
OpenSearch knn_vector index; STORE=s3vectors uses Amazon S3 Vectors. All expose
reset / add / finalize / search, so the embed step and retrieval are identical
whichever store is behind them.
"""

import os


def get_store():
    """Return the vector store selected by the STORE environment variable."""
    backend = os.environ.get("STORE", "pgvector")
    if backend == "opensearch":
        from opensearch_store import OpenSearchStore

        return OpenSearchStore()
    if backend == "s3vectors":
        from s3vectors_store import S3VectorsStore

        return S3VectorsStore()
    if backend == "pgvector":
        from pgvector_store import PgVectorStore

        return PgVectorStore()
    raise ValueError(
        f"unknown STORE={backend!r}; use pgvector, opensearch, or s3vectors"
    )
