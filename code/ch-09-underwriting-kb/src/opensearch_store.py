"""OpenSearch store: a local container, Amazon OpenSearch Service on AWS.

The store interface (reset / add / finalize / search) the chapter uses, so the
embed step and retrieval do not care which backend is behind them.
"""

import datetime
import os

from models import embed
from opensearchpy import OpenSearch, helpers

INDEX = "memo_chunks"
PIPELINE = "memo-hybrid"

SSL = os.environ.get("OPENSEARCH_SSL") == "1"
INSECURE = os.environ.get("OPENSEARCH_INSECURE") == "1"

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
    # borrower, the exact tokens an embedding blurs into its neighbors
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
        use_ssl=SSL,
        # The managed domain is https with a certificate from a public CA, and
        # the master-user password is sent on every request, so the certificate
        # is verified. Only a local endpoint with a self-signed certificate has
        # a reason to skip it, and it has to say so: OPENSEARCH_INSECURE=1.
        verify_certs=SSL and not INSECURE,
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
                        "total_inflows": {"type": "double"},
                        "existing_repayments": {"type": "double"},
                        "months": {"type": "integer"},
                        "reconciliation": {"type": "keyword"},
                        "balance_breaks": {"type": "keyword"},
                        "excluded_inflows": {"type": "keyword"},
                        # an upload replaces the one before it rather than
                        # joining it, so each submission carries its own version
                        # and a read names the version it read
                        "submission_version": {"type": "integer"},
                        "uploaded_at": {"type": "date"},
                        # supporting documents the underwriter adds to a case
                        "doc_name": {"type": "keyword"},
                        "doc_type": {"type": "keyword"},
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
        doc_id: str | None = None,
    ) -> None:
        """Buffer an embedded chunk; a supplied doc_id replaces the same indexed document."""
        self._buffer.append(
            {
                "_index": INDEX,
                **({"_id": doc_id} if doc_id else {}),
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
        vector: list[float] | None = None,
    ) -> list[tuple[int, str, str, float]]:
        """Return the k best chunks as (loan_id, borrower, content, score).

        The scope keeps a document under review out of the results, so it is
        never returned as a precedent for itself. With hybrid on, the score is the
        pipeline's combined 0 to 1 score; with it off, a cosine similarity. Pass
        the query's vector when it is already embedded, to skip embedding it again.
        """
        if vector is None:
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
    def latest_version(self, case_id: str, source: str = "submission") -> int:
        """Return the newest version of an uploaded document, or 0 if none."""
        if not self.client.indices.exists(index=INDEX):
            return 0
        resp = self.client.search(
            index=INDEX,
            body={
                "size": 0,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"source": source}},
                            {"term": {"case_id": case_id}},
                        ]
                    }
                },
                "aggs": {"latest": {"max": {"field": "submission_version"}}},
            },
        )
        return int(resp["aggregations"]["latest"]["value"] or 0)

    def index_submission(
        self,
        runtime,
        case_id: str,
        holder: str,
        chunks: list[str],
        meta: dict | None = None,
        source: str = "submission",
        doc_name: str = "",
    ) -> int:
        """Index a numbered submission version with version-specific chunk IDs."""
        version = self.latest_version(case_id, source) + 1
        uploaded_at = datetime.datetime.now(datetime.UTC).isoformat()
        vectors = embed(runtime, chunks)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            self.add(
                0,
                holder,
                i,
                chunk,
                vector,
                source=source,
                case_id=case_id,
                meta={
                    **(meta or {}),
                    "submission_version": version,
                    "uploaded_at": uploaded_at,
                    "doc_name": doc_name,
                },
                doc_id=f"{source}:{case_id}:v{version}:{i}",
            )
        self.finalize()
        return version

    def read_submission(
        self, case_id: str, version: int | None = None, source: str = "submission"
    ) -> dict:
        """Return figures and text for the requested version, or the latest version."""
        wanted = self.latest_version(case_id, source) if version is None else version
        if not wanted:
            return {}
        resp = self.client.search(
            index=INDEX,
            body={
                "size": 500,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"source": source}},
                            {"term": {"case_id": case_id}},
                            {"term": {"submission_version": wanted}},
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
            "version": wanted,
            "uploaded_at": first.get("uploaded_at", ""),
            "doc_name": first.get("doc_name", ""),
            "holder": first.get("borrower", ""),
            "monthly_income": first.get("monthly_income"),
            "total_inflows": first.get("total_inflows"),
            "existing_repayments": first.get("existing_repayments"),
            "months": first.get("months"),
            "doc_type": first.get("doc_type", ""),
            "reconciliation": first.get("reconciliation", "not_checked"),
            "balance_breaks": first.get("balance_breaks") or [],
            "excluded_inflows": first.get("excluded_inflows") or [],
            "text": "\n\n".join(h["content"] for h in hits),
        }

    def list_submissions(self, case_id: str, source: str = "context") -> list[dict]:
        """Return all matching case-document versions, oldest first."""
        if not self.client.indices.exists(index=INDEX):
            return []
        resp = self.client.search(
            index=INDEX,
            body={
                "size": 1000,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"source": source}},
                            {"term": {"case_id": case_id}},
                        ]
                    }
                },
                "sort": [{"submission_version": "asc"}, {"chunk_index": "asc"}],
            },
        )
        documents: dict[int, dict] = {}
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            version = src.get("submission_version") or 0
            doc = documents.setdefault(
                version,
                {
                    "version": version,
                    "doc_name": src.get("doc_name", ""),
                    "doc_type": src.get("doc_type", ""),
                    "uploaded_at": src.get("uploaded_at", ""),
                    "chunks": [],
                },
            )
            doc["chunks"].append(src["content"])
        ordered = [documents[v] for v in sorted(documents)]
        for doc in ordered:
            doc["text"] = "\n\n".join(doc.pop("chunks"))
        return ordered

    def drop_submission(
        self, case_id: str, version: int | None = None, source: str = "submission"
    ) -> int:
        """Delete the selected submission version, or all versions when none is specified."""
        filters: list[dict] = [
            {"term": {"source": source}},
            {"term": {"case_id": case_id}},
        ]
        if version is not None:
            filters.append({"term": {"submission_version": version}})
        resp = self.client.delete_by_query(
            index=INDEX,
            body={"query": {"bool": {"filter": filters}}},
            refresh=True,
        )
        return resp.get("deleted", 0)
