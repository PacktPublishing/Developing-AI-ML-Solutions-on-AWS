"""The vector-store seam: one interface, two backends, selected by STORE.

STORE=pgvector (default) uses Postgres with pgvector; STORE=opensearch uses an
OpenSearch knn_vector index. Both expose reset / add / finalize / search, so the
embed step and retrieval are identical whichever store is behind them.
"""

import os


def get_store():
    """Return the vector store selected by the STORE environment variable."""
    backend = os.environ.get("STORE", "pgvector")
    if backend == "opensearch":
        from opensearch_store import OpenSearchStore

        return OpenSearchStore()
    if backend == "pgvector":
        from pgvector_store import PgVectorStore

        return PgVectorStore()
    raise ValueError(f"unknown STORE={backend!r}; use pgvector or opensearch")
