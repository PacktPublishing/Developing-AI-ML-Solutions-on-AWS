# /// script
# dependencies = ["requests", "boto3", "requests-aws4auth"]
# ///
"""The neural plugin on a managed domain, where Bedrock does the embedding.

Locally the cluster hosts a small model of its own (local/neural_search.py). A
managed domain does it the other way round: ml-commons holds a connector to
Bedrock, assumes an IAM role to call it, and embeds through that at ingest time
and at query time. The model never runs on the node, which is what keeps this
possible on a t3.small.

Two identities are in play, which is the part that catches people out. The master
user is a fine-grained-access-control login and AWS sees those requests as
anonymous, so creating a connector with a roleArn fails on iam:PassRole. That call
has to be signed with SigV4 by an IAM principal instead, and that principal has to
be mapped to a security role first, which the master user can do.

Usage (from aws/):
  make neural
"""

import argparse
import sys
import time

import boto3
import requests
from requests.auth import HTTPBasicAuth
from requests_aws4auth import AWS4Auth

EMBED_MODEL = "amazon.titan-embed-text-v2:0"
DIMENSION = 1024
INDEX = "memo_chunks_neural"
PIPELINE = "memo-embed"


def _post(session, url, body=None):
    """POST JSON and raise with the response body, which carries the real reason."""
    r = session.post(url, json=body, timeout=60)
    if r.status_code >= 400:
        sys.exit(f"{url} -> {r.status_code}\n{r.text[:600]}")
    return r.json()


def _wait(session, host, task_id, label):
    """Poll one ml-commons task until it stops running."""
    for _ in range(60):
        task = session.get(f"{host}/_plugins/_ml/tasks/{task_id}", timeout=30).json()
        if task.get("state") in ("COMPLETED", "FAILED"):
            if task["state"] == "FAILED":
                sys.exit(f"{label} failed: {task.get('error')}")
            return task
        time.sleep(5)
    sys.exit(f"{label} did not finish in time")


def main() -> None:
    """Create the connector, register the model, and run one neural query."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", required=True, help="https://<domain endpoint>")
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--role-arn", required=True, help="role the domain assumes")
    p.add_argument("--region", default="us-east-1")
    p.add_argument(
        "--query", default="deferred principal while receivables were delayed"
    )
    args = p.parse_args()

    host = args.endpoint.rstrip("/")

    # the master user: fine-grained access control, and what can change security config
    admin = requests.Session()
    admin.auth = HTTPBasicAuth(args.user, args.password)

    # the IAM caller: what AWS evaluates iam:PassRole against
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    identity = boto3.client("sts", region_name=args.region).get_caller_identity()
    signed = requests.Session()
    signed.auth = AWS4Auth(
        creds.access_key,
        creds.secret_key,
        args.region,
        "es",
        session_token=creds.token,
    )

    # give the IAM caller the ml-commons permissions, or its signed requests are
    # authenticated by AWS and then refused by the security plugin
    caller_arn = identity["Arn"].replace(":sts:", ":iam:")
    caller_arn = caller_arn.replace(":assumed-role/", ":role/").rsplit("/", 1)[0]
    # PUT rather than PATCH: a PATCH here returns 200 and changes nothing
    mapped = admin.put(
        f"{host}/_plugins/_security/api/rolesmapping/ml_full_access",
        json={"backend_roles": [caller_arn]},
        timeout=30,
    )
    if mapped.status_code >= 400:
        sys.exit(f"could not map {caller_arn}: {mapped.text[:300]}")
    print(f"mapped {caller_arn} to ml_full_access")

    # the signed session is only for ml-commons, where PassRole is evaluated;
    # indices and pipelines stay with the master user, which owns them
    session = signed

    # a single node has no dedicated ML node to place the connector on
    admin.put(
        f"{host}/_cluster/settings",
        json={
            "persistent": {
                "plugins.ml_commons.only_run_on_ml_node": False,
                "plugins.ml_commons.model_access_control_enabled": False,
                "plugins.ml_commons.trusted_connector_endpoints_regex": [
                    "^https://bedrock-runtime\\..*[a-z0-9-]\\.amazonaws\\.com/.*$"
                ],
            }
        },
        timeout=30,
    ).raise_for_status()

    connector = _post(
        session,
        f"{host}/_plugins/_ml/connectors/_create",
        {
            "name": "bedrock titan embeddings",
            "description": "embeddings for the memo knowledge base",
            "version": 1,
            "protocol": "aws_sigv4",
            "credential": {"roleArn": args.role_arn},
            "parameters": {
                "region": args.region,
                "service_name": "bedrock",
                "model": EMBED_MODEL,
                "dimensions": DIMENSION,
                "normalize": True,
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": f"https://bedrock-runtime.{args.region}.amazonaws.com/model/{EMBED_MODEL}/invoke",
                    "headers": {"content-type": "application/json"},
                    "request_body": (
                        '{"inputText": "${parameters.inputText}", '
                        '"dimensions": ${parameters.dimensions}, '
                        '"normalize": ${parameters.normalize}}'
                    ),
                    "pre_process_function": "connector.pre_process.bedrock.embedding",
                    "post_process_function": "connector.post_process.bedrock.embedding",
                }
            ],
        },
    )
    connector_id = connector["connector_id"]
    print(f"connector: {connector_id}")

    registered = _post(
        session,
        f"{host}/_plugins/_ml/models/_register",
        {
            "name": "bedrock-titan-embed",
            "function_name": "remote",
            "description": "Titan text embeddings through the connector",
            "connector_id": connector_id,
        },
    )
    model_id = _wait(session, host, registered["task_id"], "register")["model_id"]
    deployed = _post(session, f"{host}/_plugins/_ml/models/{model_id}/_deploy")
    _wait(session, host, deployed["task_id"], "deploy")
    print(f"model: {model_id}")

    admin.put(
        f"{host}/_ingest/pipeline/{PIPELINE}",
        json={
            "description": "embed memo chunks through Bedrock as they are indexed",
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

    admin.delete(f"{host}/{INDEX}", timeout=60)
    admin.put(
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
    ).raise_for_status()

    memos = [
        (
            40118226,
            "ADAEZE FABRICS LIMITED",
            (
                "Kindly approve a two-month principal moratorium. Receivables from "
                "two distributors now fall due in April rather than February."
            ),
        ),
        (
            40118301,
            "OKON LOGISTICS",
            (
                "Requesting an increase of 4,000,000 on the existing facility. "
                "Combined exposure stays inside the single-obligor limit."
            ),
        ),
    ]
    for loan_id, borrower, content in memos:
        admin.post(
            f"{host}/{INDEX}/_doc",
            json={"loan_id": loan_id, "borrower": borrower, "content": content},
            timeout=60,
        ).raise_for_status()
    admin.post(f"{host}/{INDEX}/_refresh", timeout=30)

    found = _post(
        admin,
        f"{host}/{INDEX}/_search",
        {
            "size": 2,
            "_source": ["loan_id", "content"],
            "query": {
                "neural": {
                    "embedding": {
                        "query_text": args.query,
                        "model_id": model_id,
                        "k": 2,
                    }
                }
            },
        },
    )
    print(f"\nneural query ({found['took']}ms): {args.query}")
    for hit in found["hits"]["hits"]:
        src = hit["_source"]
        print(f"  {hit['_score']:.4f}  [{src['loan_id']}]  {src['content'][:62]}")


if __name__ == "__main__":
    main()
