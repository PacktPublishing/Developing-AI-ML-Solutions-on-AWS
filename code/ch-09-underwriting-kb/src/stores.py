"""The vector-store seam: one interface, one backend.

The store is Amazon OpenSearch Service, an OpenSearch container locally and a
managed domain on AWS. The seam stays a function rather than an import so the
embed step and the retrieval code name the store in one place, and so a reader
swapping in another backend has one thing to change.
"""


def get_store():
    """Return the vector store the chapter uses."""
    from opensearch_store import OpenSearchStore

    return OpenSearchStore()
