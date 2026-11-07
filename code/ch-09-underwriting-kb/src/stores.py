"""Create the OpenSearch store used by ingestion and retrieval."""


def get_store():
    """Return the vector store the chapter uses."""
    from opensearch_store import OpenSearchStore

    return OpenSearchStore()
