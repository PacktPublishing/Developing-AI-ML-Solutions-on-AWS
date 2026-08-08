# /// script
# dependencies = ["requests", "requests-aws4auth", "boto3"]
# ///
"""Provision OpenSearch conversational-search RAG over Bedrock for Dashboards.

Reading the ml-commons source (search-processors DefaultLlmImpl) is what makes this
work: when llm_response_field is set, the retrieval_augmented_generation processor
sends the built prompt as ${parameters.inputs} (not "prompt" or "messages"), and it
reads the answer with a top-level dataAsMap.get(field). So the connector maps
${parameters.inputs} into Bedrock Converse, and a painless post_process (flatten.painless)
lifts the nested Converse answer to a top-level "response" key. Connector/model
creation are SigV4-signed by the deploy role (needs iam:PassRole + an ml_full_access
mapping); the pipeline and query use the master user (all_access). Needs OpenSearch
2.13+ (the RAG feature is off on 2.11 and not settable on the managed service).

Usage (env: ENDPOINT, OS_USER, OS_PASS, RAG_ROLE_ARN, CH09_USER_ARN, REGION):
  uv run provision.py all      # map -> connector -> model -> pipeline -> query
"""

import json
import os
import sys
import time
from pathlib import Path

import boto3
import requests
from requests_aws4auth import AWS4Auth

EP = os.environ["ENDPOINT"]
MASTER = (os.environ["OS_USER"], os.environ["OS_PASS"])
ROLE = os.environ["RAG_ROLE_ARN"]
CH09_USER_ARN = os.environ["CH09_USER_ARN"]
REGION = os.environ.get("REGION", "us-east-1")
MODEL = os.environ.get("MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
BASE = f"https://{EP}"
PAINLESS = (Path(__file__).parent / "flatten.painless").read_text()

_c = boto3.Session().get_credentials()
SIG = AWS4Auth(_c.access_key, _c.secret_key, REGION, "es", session_token=_c.token)


def call(method: str, path: str, body: dict | None = None, auth: str = "sig") -> dict:
    """Send a signed (SigV4) or master (basic-auth) request; print status; return JSON."""
    r = requests.request(
        method,
        BASE + path,
        auth=(SIG if auth == "sig" else MASTER),
        data=json.dumps(body) if body else None,
        headers={"Content-Type": "application/json"},
        timeout=90,
    )
    print(f"{method} {path} -> {r.status_code}")
    return r.json() if r.text else {}


def register_model(connector_id: str) -> str:
    """Register + deploy a remote model on the connector, retrying past ml-commons throttles."""
    for _ in range(6):
        reg = call(
            "POST",
            "/_plugins/_ml/models/_register",
            {
                "name": "bedrock-claude-rag",
                "function_name": "remote",
                "connector_id": connector_id,
                "deploy": True,
            },
        )
        if reg.get("model_id"):
            return reg["model_id"]
        time.sleep(8)
    raise RuntimeError("model register kept throttling")


def main() -> None:
    """Provision the RAG stack end to end and run one grounded query."""
    # 1. FGAC: let the deploy role create connectors/models
    call(
        "PUT",
        "/_plugins/_security/api/rolesmapping/ml_full_access",
        {"backend_roles": [CH09_USER_ARN], "users": [MASTER[0]]},
        auth="master",
    )

    # 2. connector: Bedrock Converse; ${parameters.inputs} is the prompt the RAG
    #    processor sends; the painless post_process lifts the answer to "response".
    conn = call(
        "POST",
        "/_plugins/_ml/connectors/_create",
        {
            "name": "Bedrock Claude (RAG)",
            "version": 1,
            "protocol": "aws_sigv4",
            "credential": {"roleArn": ROLE},
            "parameters": {"region": REGION, "service_name": "bedrock", "model": MODEL},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{MODEL}/converse",
                    "headers": {"content-type": "application/json"},
                    "request_body": '{ "messages": [{"role":"user","content":[{"text":"${parameters.inputs}"}]}], "inferenceConfig": {"maxTokens": 1000, "temperature": 0} }',
                    "post_process_function": PAINLESS,
                }
            ],
        },
    )
    model_id = register_model(conn["connector_id"])
    call("POST", f"/_plugins/_ml/models/{model_id}/_deploy")
    time.sleep(6)
    print("model", model_id)

    # 3. the RAG search pipeline
    call(
        "PUT",
        "/_search/pipeline/rag_pipeline",
        auth="master",
        body={
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

    # 4. a grounded query -- the answer comes back in the response ext
    q = call(
        "POST",
        "/memo_chunks/_search?search_pipeline=rag_pipeline",
        auth="master",
        body={
            "query": {"match": {"content": "grocery DTI inflows"}},
            "size": 4,
            "_source": ["loan_id", "borrower", "content"],
            "ext": {
                "generative_qa_parameters": {
                    "llm_model": "bedrock/claude",
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
