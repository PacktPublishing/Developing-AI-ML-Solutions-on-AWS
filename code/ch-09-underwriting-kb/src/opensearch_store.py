"""OpenSearch store: a local container, Amazon OpenSearch Service on AWS.

The store interface (reset / add / finalize / search) the chapter uses, so the
embed step and retrieval do not care which backend is behind them.
"""

import os

from models import embed
from opensearchpy import OpenSearch, helpers

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
        # a small managed node throttles under a burst of writes; ride out the 429s
        timeout=30,
        max_retries=5,
        retry_on_status=(429, 502, 503, 504),
        retry_on_timeout=True,
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
                        # corpus chunks are past decisions; submission chunks are the
                        # document under review, kept apart so one cannot cite the other
                        "source": {"type": "keyword"},
                        "case_id": {"type": "keyword"},
                        # the figures the statement parser recovered, carried with
                        # the document so the agent reads them back from the index
                        "monthly_income": {"type": "double"},
                        "existing_repayments": {"type": "double"},
                        "months": {"type": "integer"},
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
        source: str = "corpus",
        case_id: str = "",
        meta: dict | None = None,
    ) -> None:
        """Buffer one embedded chunk for bulk indexing."""
        self._buffer.append(
            {
                "_index": INDEX,
                "source": source,
                "case_id": case_id,
                **(meta or {}),
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
            # small chunks with backoff so a modest node is not overrun by one big bulk
            helpers.bulk(
                self.client,
                self._buffer,
                chunk_size=100,
                max_retries=4,
                initial_backoff=2,
                max_backoff=30,
                request_timeout=60,
            )
            self._buffer = []
        self.client.indices.refresh(index=INDEX)

    def search(
        self, runtime, query: str, k: int = 5, source: str = "corpus"
    ) -> list[tuple[int, str, str, float]]:
        """Return the k nearest chunks as (loan_id, borrower, content, similarity).

        The filter defaults to the corpus so a document under review is never
        returned as a precedent for itself.
        """
        vector = embed(runtime, [query])[0]
        knn = {"vector": vector, "k": k}
        if source:
            knn["filter"] = {"term": {"source": source}}
        resp = self.client.search(
            index=INDEX,
            body={"size": k, "query": {"knn": {"embedding": knn}}},
        )
        hits = []
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            # lucene cosinesimil maps cosine c to score (1 + c) / 2, so invert it
            # and hand the caller a cosine similarity on its own scale
            similarity = 2 * h["_score"] - 1
            hits.append((src["loan_id"], src["borrower"], src["content"], similarity))
        return hits

    # ---------------------------------------------------------------------------
    # The document under review
    # ---------------------------------------------------------------------------
    def index_submission(
        self,
        runtime,
        case_id: str,
        holder: str,
        chunks: list[str],
        meta: dict | None = None,
    ) -> int:
        """Embed and index one uploaded document, with the figures parsed from it."""
        vectors = embed(runtime, chunks)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            self.add(
                0,
                holder,
                i,
                chunk,
                vector,
                source="submission",
                case_id=case_id,
                meta=meta,
            )
        self.finalize()
        return len(chunks)

    def read_submission(self, case_id: str) -> dict:
        """Return the uploaded document: its parsed figures and its text."""
        resp = self.client.search(
            index=INDEX,
            body={
                "size": 100,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"source": "submission"}},
                            {"term": {"case_id": case_id}},
                        ]
                    }
                },
                "sort": [{"chunk_index": "asc"}],
            },
        )
        hits = [h["_source"] for h in resp["hits"]["hits"]]
        if not hits:
            return {}
        first = hits[0]
        return {
            "case_id": case_id,
            "holder": first.get("borrower", ""),
            "monthly_income": first.get("monthly_income"),
            "existing_repayments": first.get("existing_repayments"),
            "months": first.get("months"),
            "text": "\n\n".join(h["content"] for h in hits),
        }

    def drop_submission(self, case_id: str) -> int:
        """Delete an uploaded document once the review is done."""
        resp = self.client.delete_by_query(
            index=INDEX,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"source": "submission"}},
                            {"term": {"case_id": case_id}},
                        ]
                    }
                }
            },
            refresh=True,
        )
        return resp.get("deleted", 0)
