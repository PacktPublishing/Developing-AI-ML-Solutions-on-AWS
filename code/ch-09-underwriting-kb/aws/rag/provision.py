# /// script
# dependencies = ["requests", "requests-aws4auth", "boto3"]
# ///
"""Provision OpenSearch conversational-search RAG over Bedrock, step by step.

Registers a Bedrock Claude connector as an ml-commons remote model and builds a
retrieval_augmented_generation search pipeline, so Dashboards can answer questions
grounded in the memo index. Connector and model creation are SigV4-signed by the
deploy role (needs iam:PassRole on the connector role and an ml_full_access
mapping); the pipeline and query use the master user (all_access). The pipeline
step needs plugins.ml_commons.rag_pipeline_feature_enabled, which is on by default
on OpenSearch 2.13+ but off on 2.11 (and not settable on the managed service).

Usage (env: ENDPOINT, OS_USER, OS_PASS, RAG_ROLE_ARN, CH09_USER_ARN, REGION):
  uv run provision.py map                       # master maps deploy role -> ml_full_access
  uv run provision.py connector                 # -> CONNECTOR_ID
  uv run provision.py model <connector_id>      # register+deploy -> MODEL_ID
  uv run provision.py deploy <model_id>         # ensure deployed
  uv run provision.py predict <model_id>        # smoke test: Claude generates
  uv run provision.py pipeline <model_id>       # needs the RAG feature (2.13+)
  uv run provision.py query                      # grounded answer in the response ext
"""

import json
import os
import sys
import time

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

_c = boto3.Session().get_credentials()
SIGV4 = AWS4Auth(_c.access_key, _c.secret_key, REGION, "es", session_token=_c.token)


def call(method: str, path: str, body: dict | None = None, auth: str = "sig") -> dict:
    """Send a signed (SigV4) or master (basic-auth) request and print the result."""
    a = SIGV4 if auth == "sig" else MASTER
    r = requests.request(
        method,
        BASE + path,
        auth=a,
        data=json.dumps(body) if body else None,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    try:
        j = r.json()
    except ValueError:
        j = {"text": r.text}
    print(f"{method} {path} -> {r.status_code}")
    print(json.dumps(j, indent=2)[:1400])
    print("-" * 60)
    return j


def _wait_task(task_id: str | None) -> dict:
    """Poll an ml-commons task until it completes or fails."""
    if not task_id:
        return {}
    for _ in range(30):
        t = call("GET", f"/_plugins/_ml/tasks/{task_id}")
        if t.get("state") in ("COMPLETED", "FAILED"):
            return t
        time.sleep(4)
    return {}


def main() -> None:
    """Run one provisioning step named on the command line."""
    step = sys.argv[1]

    if step == "map":
        call(
            "PUT",
            "/_plugins/_security/api/rolesmapping/ml_full_access",
            {"backend_roles": [CH09_USER_ARN], "users": [MASTER[0]]},
            auth="master",
        )

    elif step == "connector":
        conn = call(
            "POST",
            "/_plugins/_ml/connectors/_create",
            {
                "name": "Bedrock Claude (RAG)",
                "description": "Claude on Bedrock via Converse",
                "version": 1,
                "protocol": "aws_sigv4",
                "credential": {"roleArn": ROLE},
                "parameters": {
                    "region": REGION,
                    "service_name": "bedrock",
                    "model": MODEL,
                    "max_tokens": 1000,
                    "temperature": 0.0,
                },
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{MODEL}/converse",
                        "headers": {"content-type": "application/json"},
                        "request_body": '{ "messages": [{"role":"user","content":[{"text":"${parameters.prompt}"}]}], "inferenceConfig": {"maxTokens": ${parameters.max_tokens}, "temperature": ${parameters.temperature}} }',
                    }
                ],
            },
        )
        print("CONNECTOR_ID", conn.get("connector_id"))

    elif step == "model":
        grp = call(
            "POST",
            "/_plugins/_ml/model_groups/_register",
            {"name": "rag-claude", "description": "RAG Claude"},
        )
        reg = call(
            "POST",
            "/_plugins/_ml/models/_register",
            {
                "name": "bedrock-claude-rag",
                "function_name": "remote",
                "model_group_id": grp.get("model_group_id"),
                "connector_id": sys.argv[2],
                "deploy": True,
            },
        )
        t = _wait_task(reg.get("task_id"))
        print("MODEL_ID", t.get("model_id"), "STATE", t.get("state"))

    elif step == "deploy":
        d = call("POST", f"/_plugins/_ml/models/{sys.argv[2]}/_deploy")
        print("DEPLOY STATE", _wait_task(d.get("task_id")).get("state"))

    elif step == "predict":
        call(
            "POST",
            f"/_plugins/_ml/models/{sys.argv[2]}/_predict",
            {"parameters": {"prompt": "Say 'connector works' and nothing else."}},
        )

    elif step == "pipeline":
        call(
            "PUT",
            "/_search/pipeline/rag_pipeline",
            auth="master",
            body={
                "response_processors": [
                    {
                        "retrieval_augmented_generation": {
                            "tag": "memo_rag",
                            "description": "grounded answers over memo_chunks",
                            "model_id": sys.argv[2],
                            "context_field_list": ["content"],
                            "system_prompt": "You are a credit underwriting assistant. Answer only from the context, and cite the source loan id for each claim.",
                            "user_instructions": "Answer the question grounded in the memo passages.",
                        }
                    }
                ]
            },
        )

    elif step == "query":
        call(
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
                        "llm_question": "How is DTI assessed for a grocery business, and what inflows were seen?",
                        "context_size": 4,
                        "timeout": 30,
                    }
                },
            },
        )


if __name__ == "__main__":
    main()
