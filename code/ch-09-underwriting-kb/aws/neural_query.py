# /// script
# dependencies = ["requests"]
# ///
"""Query the neural index, where the domain embeds instead of the caller.

The connector, the model and the ingest pipeline are infrastructure: the stack
builds them (see NeuralModel in template.yaml) and reports the model id. What is
left here is data-plane work, an index that runs the pipeline by default and one
query that carries text rather than a vector.

Usage (from aws/):
  make neural
"""

import argparse
import sys

import requests
from requests.auth import HTTPBasicAuth

DIMENSION = 1024
INDEX = "memo_chunks_neural"
PIPELINE = "memo-embed"

MEMOS = [
    (
        40118226,
        "ADAEZE FABRICS LIMITED",
        (
            "Kindly approve a two-month principal moratorium. Receivables from two "
            "distributors now fall due in April rather than February."
        ),
    ),
    (
        40118301,
        "OKON LOGISTICS",
        (
            "Requesting an increase of 4,000,000 on the existing facility. Combined "
            "exposure stays inside the single-obligor limit."
        ),
    ),
]


def main() -> None:
    """Index a few memos through the pipeline and run one neural query."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--model-id", required=True, help="from the stack's NeuralModelId")
    p.add_argument(
        "--query", default="deferred principal while receivables were delayed"
    )
    args = p.parse_args()
    if not args.model_id or not args.endpoint.strip():
        sys.exit("no model id or endpoint: is the stack deployed?")

    host = args.endpoint.rstrip("/")
    session = requests.Session()
    session.auth = HTTPBasicAuth(args.user, args.password)

    session.delete(f"{host}/{INDEX}", timeout=60)
    created = session.put(
        f"{host}/{INDEX}",
        json={
            "settings": {"index.knn": True, "default_pipeline": PIPELINE},
            "mappings": {
                "properties": {
                    "loan_id": {"type": "long"},
                    "borrower": {"type": "keyword"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": DIMENSION,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                        },
                    },
                }
            },
        },
        timeout=60,
    )
    if created.status_code >= 400:
        sys.exit(f"could not create {INDEX}: {created.text[:400]}")

    # no vector is sent here: the pipeline embeds each document on the way in
    for loan_id, borrower, content in MEMOS:
        stored = session.post(
            f"{host}/{INDEX}/_doc",
            json={"loan_id": loan_id, "borrower": borrower, "content": content},
            timeout=60,
        )
        if stored.status_code >= 400:
            sys.exit(f"ingest failed: {stored.text[:400]}")
    session.post(f"{host}/{INDEX}/_refresh", timeout=30)

    # and none here either: the query carries the question as text
    found = session.post(
        f"{host}/{INDEX}/_search",
        json={
            "size": 2,
            "_source": ["loan_id", "content"],
            "query": {
                "neural": {
                    "embedding": {
                        "query_text": args.query,
                        "model_id": args.model_id,
                        "k": 2,
                    }
                }
            },
        },
        timeout=60,
    )
    if found.status_code >= 400:
        sys.exit(f"neural query failed: {found.text[:400]}")
    body = found.json()

    print(f"neural query ({body['took']}ms): {args.query}")
    for hit in body["hits"]["hits"]:
        src = hit["_source"]
        print(f"  {hit['_score']:.4f}  [{src['loan_id']}]  {src['content'][:62]}")


if __name__ == "__main__":
    main()
