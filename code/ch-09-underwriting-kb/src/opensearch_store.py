"""OpenSearch store: a local container, Amazon OpenSearch Service on AWS.

The store interface (reset / add / finalize / search) the chapter uses, so the
embed step and retrieval do not care which backend is behind them.
"""

import os

from models import embed
from opensearchpy import OpenSearch, helpers

INDEX = "memo_chunks"
PIPELINE = "memo-hybrid"

# What a search reads by default: the decided cases and the credit policy, never
# a document under review.
ARCHIVE = ("corpus", "policy")


def scope(filters: dict | None = None, sources: tuple[str, ...] = ARCHIVE) -> dict:
    """Return the filter a search runs under.

    Policy applies to every deal, so it is always in scope. The filters narrow
    only the precedent: sector, an exposure range, and how recent the decision is.
    """
    filters = filters or {}
    cases: list[dict] = [{"terms": {"source": [s for s in sources if s != "policy"]}}]
    if filters.get("sector"):
        cases.append({"term": {"sector": filters["sector"]}})
    exposure = {
        op: filters[key]
        for op, key in (("gte", "min_exposure"), ("lte", "max_exposure"))
        if filters.get(key) is not None
    }
    if exposure:
        cases.append({"range": {"exposure_usd": exposure}})
    if filters.get("since"):
        cases.append({"range": {"decided": {"gte": filters["since"]}}})
    should: list[dict] = [{"bool": {"filter": cases}}]
    if "policy" in sources:
        should.append({"term": {"source": "policy"}})
    return {"bool": {"should": should, "minimum_should_match": 1}}


def query_body(
    vector: list[float], text: str, k: int, where: dict, hybrid: bool
) -> dict:
    """Return the search body: k-NN alone, or k-NN beside a keyword match."""
    knn = {"knn": {"embedding": {"vector": vector, "k": k, "filter": where}}}
    if not hybrid:
        return {"size": k, "query": knn}
    # the keyword half reads the title too, which carries the loan id and the
    # borrower, the exact tokens an embedding blurs into its neighbours
    keyword = {
        "bool": {
            "must": {"multi_match": {"query": text, "fields": ["title^2", "content"]}},
            "filter": where,
        }
    }
    return {"size": k, "query": {"hybrid": {"queries": [keyword, knn]}}}


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
        # the search pipeline puts the BM25 and cosine scores on one scale before
        # combining them. Equal weights: at 0.3 for the keyword half a loan id was
        # outvoted by passages that merely read alike (make eval shows the gap)
        self.client.transport.perform_request(
            "PUT",
            f"/_search/pipeline/{PIPELINE}",
            body={
                "phase_results_processors": [
                    {
                        "normalization-processor": {
                            "normalization": {"technique": "min_max"},
                            "combination": {
                                "technique": "arithmetic_mean",
                                "parameters": {"weights": [0.5, 0.5]},
                            },
                        }
                    }
                ]
            },
        )
        self.client.indices.create(
            index=INDEX,
            body={
                # one shard suits a corpus this size, and keeps scores the same on the
                # local node and the managed domain, which defaults to five
                "settings": {"index.knn": True, "number_of_shards": 1},
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
                        "balance_breaks": {"type": "keyword"},
                        "loan_id": {"type": "long"},
                        "borrower": {"type": "keyword"},
                        # the loan-book fields the filters narrow on
                        "sector": {"type": "keyword"},
                        "exposure_usd": {"type": "long"},
                        "tenor_months": {"type": "integer"},
                        "decided": {"type": "date"},
                        # loan id, borrower, and document name as searchable text
                        "title": {"type": "text"},
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
                "title": f"{loan_id or ''} {borrower}".strip(),
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
        self,
        runtime,
        query: str,
        k: int = 5,
        filters: dict | None = None,
        hybrid: bool = True,
        sources: tuple[str, ...] = ARCHIVE,
    ) -> list[tuple[int, str, str, float]]:
        """Return the k best chunks as (loan_id, borrower, content, score).

        The scope keeps a document under review out of the results, so it is
        never returned as a precedent for itself. With hybrid on, the score is the
        pipeline's combined 0 to 1 score; with it off, a cosine similarity.
        """
        vector = embed(runtime, [query])[0]
        body = query_body(vector, query, k, scope(filters, sources), hybrid)
        params = {"search_pipeline": PIPELINE} if hybrid else {}
        resp = self.client.search(index=INDEX, body=body, params=params)
        hits = []
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            # lucene cosinesimil maps cosine c to score (1 + c) / 2, so invert it
            # and hand the caller a cosine similarity on its own scale
            score = h["_score"] if hybrid else 2 * h["_score"] - 1
            hits.append((src["loan_id"], src["borrower"], src["content"], score))
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
            "balance_breaks": first.get("balance_breaks") or [],
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
