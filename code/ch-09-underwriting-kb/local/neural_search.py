# /// script
# dependencies = ["requests"]
# ///
"""Prove the neural plugin: let the cluster embed, instead of embedding first.

Everywhere else in this chapter the app calls the model and sends OpenSearch a
vector. Here OpenSearch holds the model: an ingest pipeline embeds each document
as it lands, and a neural query embeds the question at search time. The model is
a pretrained sentence-transformer running inside the cluster, so this needs no
Bedrock and no network beyond the one-off model download.

Usage:
  make neural           # against the local node on 9200
"""

import argparse
import sys
import time

import requests

MODEL_NAME = "huggingface/sentence-transformers/all-MiniLM-L6-v2"
MODEL_VERSION = "1.0.1"
DIMENSION = 384
INDEX = "memo_chunks_neural"
PIPELINE = "memo-embed"

MEMOS = [
    (
        40118226,
        "ADAEZE FABRICS LIMITED",
        (
            "Kindly approve a two-month principal moratorium. Receivables from two "
            "distributors now fall due in April rather than February. Interest "
            "continues to accrue."
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
    (
        40118455,
        "NNAMDI GROCERS",
        (
            "Approval to change the repayment account to another bank. New direct "
            "debit mandate confirmed by Risk before the next cycle."
        ),
    ),
]


def _wait(host: str, task_id: str, label: str) -> dict:
    """Poll one ml-commons task until it stops running."""
    for _ in range(60):
        task = requests.get(f"{host}/_plugins/_ml/tasks/{task_id}", timeout=30).json()
        state = task.get("state")
        if state in ("COMPLETED", "FAILED"):
            if state == "FAILED":
                sys.exit(f"{label} failed: {task.get('error')}")
            return task
        time.sleep(10)
    sys.exit(f"{label} did not finish in time")


def ensure_model(host: str) -> str:
    """Register and deploy the in-cluster embedding model, and return its id."""
    # a single-node dev cluster has no dedicated ML node, and the model is
    # pretrained rather than access-controlled
    requests.put(
        f"{host}/_cluster/settings",
        json={
            "persistent": {
                "plugins.ml_commons.only_run_on_ml_node": False,
                "plugins.ml_commons.model_access_control_enabled": False,
                "plugins.ml_commons.native_memory_threshold": 99,
            }
        },
        timeout=30,
    ).raise_for_status()

    registered = requests.post(
        f"{host}/_plugins/_ml/models/_register",
        json={
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "model_format": "TORCH_SCRIPT",
        },
        timeout=30,
    ).json()
    model_id = _wait(host, registered["task_id"], "register")["model_id"]

    deployed = requests.post(
        f"{host}/_plugins/_ml/models/{model_id}/_deploy", timeout=30
    ).json()
    _wait(host, deployed["task_id"], "deploy")
    return model_id


def ensure_index(host: str, model_id: str) -> None:
    """Create the ingest pipeline and an index that runs it by default."""
    requests.put(
        f"{host}/_ingest/pipeline/{PIPELINE}",
        json={
            "description": "embed memo chunks as they are indexed",
            "processors": [
                {
                    "text_embedding": {
                        "model_id": model_id,
                        "field_map": {"content": "embedding"},
                    }
                }
            ],
        },
        timeout=30,
    ).raise_for_status()

    requests.delete(f"{host}/{INDEX}", timeout=30)
    requests.put(
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
        timeout=30,
    ).raise_for_status()


def main() -> None:
    """Register the model, index memo text, and run one neural query."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="http://localhost:9200")
    p.add_argument(
        "--query",
        default="can we defer principal for a distributor waiting on receivables",
    )
    args = p.parse_args()

    model_id = ensure_model(args.host)
    print(f"model deployed in the cluster: {model_id}")
    ensure_index(args.host, model_id)

    # no vector is sent: the pipeline embeds each document on the way in
    for loan_id, borrower, content in MEMOS:
        requests.post(
            f"{args.host}/{INDEX}/_doc",
            json={"loan_id": loan_id, "borrower": borrower, "content": content},
            timeout=60,
        ).raise_for_status()
    requests.post(f"{args.host}/{INDEX}/_refresh", timeout=30)

    # and none is sent here either: the query carries the question as text
    found = requests.post(
        f"{args.host}/{INDEX}/_search",
        json={
            "size": 3,
            "_source": ["loan_id", "borrower", "content"],
            "query": {
                "neural": {
                    "embedding": {
                        "query_text": args.query,
                        "model_id": model_id,
                        "k": 3,
                    }
                }
            },
        },
        timeout=60,
    ).json()

    print(f"\nneural query ({found['took']}ms): {args.query}")
    for hit in found["hits"]["hits"]:
        source = hit["_source"]
        print(f"  {hit['_score']:.4f}  [{source['loan_id']}]  {source['content'][:64]}")


if __name__ == "__main__":
    main()
