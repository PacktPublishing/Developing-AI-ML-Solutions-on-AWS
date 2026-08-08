"""OpenSearch store: a local container, Amazon OpenSearch Service on AWS.

Same interface as the pgvector store (reset / add / finalize / search) so the
embed step and retrieval do not care which backend is behind them.
"""

import os

from opensearchpy import OpenSearch, helpers

from models import embed

INDEX = "memo_chunks"


def client() -> OpenSearch:
    """Return an OpenSearch client from the OPENSEARCH_* environment variables."""
    return OpenSearch(
        hosts=[
            {
                "host": os.environ.get("OPENSEARCH_HOST", "localhost"),
                "port": int(os.environ.get("OPENSEARCH_PORT", "9200")),
            }
        ],
        http_auth=(
            (os.environ["OPENSEARCH_USER"], os.environ["OPENSEARCH_PASSWORD"])
            if os.environ.get("OPENSEARCH_USER")
            else None
        ),
        use_ssl=os.environ.get("OPENSEARCH_SSL") == "1",
        verify_certs=False,
        ssl_show_warn=False,
    )


class OpenSearchStore:
    """memo_chunks as a knn_vector index: cosine k-NN over the memo corpus."""

    def __init__(self) -> None:
        """Open a client and start an empty bulk buffer."""
        self.client = client()
        self._buffer: list[dict] = []

    def reset(self, dim: int) -> None:
        """Drop and recreate the knn_vector index at width `dim`."""
        if self.client.indices.exists(index=INDEX):
            self.client.indices.delete(index=INDEX)
        self.client.indices.create(
            index=INDEX,
            body={
                "settings": {"index.knn": True},
                "mappings": {
                    "properties": {
                        "loan_id": {"type": "long"},
                        "borrower": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dim,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                            },
                        },
                    }
                },
            },
        )

    def add(
        self,
        loan_id: int,
        borrower: str,
        chunk_index: int,
        content: str,
        vector: list[float],
    ) -> None:
        """Buffer one embedded chunk for bulk indexing."""
        self._buffer.append(
            {
                "_index": INDEX,
                "loan_id": loan_id,
                "borrower": borrower,
                "chunk_index": chunk_index,
                "content": content,
                "embedding": vector,
            }
        )

    def finalize(self) -> None:
        """Flush the bulk buffer and refresh the index so it is searchable."""
        if self._buffer:
            helpers.bulk(self.client, self._buffer)
            self._buffer = []
        self.client.indices.refresh(index=INDEX)

    def search(
        self, runtime, query: str, k: int = 5
    ) -> list[tuple[int, str, str, float]]:
        """Return the k nearest chunks as (loan_id, borrower, content, similarity)."""
        vector = embed(runtime, [query])[0]
        resp = self.client.search(
            index=INDEX,
            body={
                "size": k,
                "query": {"knn": {"embedding": {"vector": vector, "k": k}}},
            },
        )
        hits = []
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            # lucene cosinesimil maps cosine c to score (1 + c) / 2; invert to a
            # cosine similarity so the number matches the pgvector store's
            similarity = 2 * h["_score"] - 1
            hits.append((src["loan_id"], src["borrower"], src["content"], similarity))
        return hits
