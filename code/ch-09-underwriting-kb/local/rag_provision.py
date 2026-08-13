# /// script
# dependencies = ["requests"]
# ///
"""Local OpenSearch conversational-search RAG over Ollama, parity of the AWS path.

Same retrieval_augmented_generation search pipeline; the connector points at the
local model instead of Bedrock.

The one difference from AWS: the local OpenSearch has security off (no SigV4, no
FGAC mapping), and Ollama's /api/generate returns the answer at a top-level
"response" key, so no painless post_process is needed -- llm_response_field just
reads "response". The RAG feature is on by default on OpenSearch 2.13+.

Usage (env OPENSEARCH_HOST defaults to localhost:9200; Ollama on the host):
  uv run local/rag_provision.py all
"""

import json
import os
import sys
import time

import requests

BASE = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
OLLAMA = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
MODEL = os.environ.get("LOCAL_MODEL", "qwen3:0.6b")


def call(method: str, path: str, body: dict | None = None) -> dict:
    """Send a request to the local OpenSearch; print status; return JSON."""
    r = requests.request(
        method,
        BASE + path,
        data=json.dumps(body) if body else None,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    print(f"{method} {path} -> {r.status_code}")
    return r.json() if r.text else {}


def main() -> None:
    """Provision the local RAG stack end to end and run one grounded query."""
    # ml-commons on a single node: run on the data node, and allow the local host
    call(
        "PUT",
        "/_cluster/settings",
        {
            "persistent": {
                "plugins.ml_commons.only_run_on_ml_node": False,
                "plugins.ml_commons.connector.private_ip_enabled": True,
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    "^http://host\\.docker\\.internal:.*$"
                ],
            }
        },
    )

    # connector to Ollama's /api/generate; ${parameters.inputs} is the prompt the
    # RAG processor sends, and the answer is already a top-level "response" key.
    conn = call(
        "POST",
        "/_plugins/_ml/connectors/_create",
        {
            "name": "Ollama (RAG)",
            "version": 1,
            "protocol": "http",
            "parameters": {"model": MODEL},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": OLLAMA,
                    "headers": {"Content-Type": "application/json"},
                    "request_body": '{ "model": "${parameters.model}", "prompt": "${parameters.inputs}", "stream": false, "think": false }',
                }
            ],
        },
    )
    reg = call(
        "POST",
        "/_plugins/_ml/models/_register",
        {
            "name": "ollama-rag",
            "function_name": "remote",
            "connector_id": conn["connector_id"],
            "deploy": True,
        },
    )
    model_id = reg["model_id"]
    call("POST", f"/_plugins/_ml/models/{model_id}/_deploy")
    time.sleep(6)
    print("model", model_id)

    call(
        "PUT",
        "/_search/pipeline/rag_pipeline",
        {
            "response_processors": [
                {
                    "retrieval_augmented_generation": {
                        "tag": "memo_rag",
                        "model_id": model_id,
                        "context_field_list": ["content"],
                        "system_prompt": "You are a credit underwriting assistant. Answer only from the memo context and cite the source loan id for each claim.",
                        "user_instructions": "Answer grounded in the memo passages.",
                    }
                }
            ]
        },
    )

    q = call(
        "POST",
        "/memo_chunks/_search?search_pipeline=rag_pipeline",
        {
            "query": {"match": {"content": "grocery DTI inflows"}},
            "size": 4,
            "_source": ["loan_id", "borrower", "content"],
            "ext": {
                "generative_qa_parameters": {
                    "llm_model": MODEL,
                    "llm_question": sys.argv[1]
                    if len(sys.argv) > 1
                    else "How is DTI assessed for a grocery business?",
                    "context_size": 4,
                    "timeout": 120,
                    "llm_response_field": "response",
                }
            },
        },
    )
    rag = q.get("ext", {}).get("retrieval_augmented_generation", {})
    print("\nANSWER:", rag.get("answer") or rag.get("error"))


if __name__ == "__main__":
    main()
