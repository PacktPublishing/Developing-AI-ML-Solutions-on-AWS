"""The underwriting assistant as a Lambda, self-contained and re-runnable.

On the first call it creates the pgvector extension, the table, and the index,
then seeds the synthetic knowledge base. The seed only runs when the table is
empty, so redeploys and repeat calls are idempotent. After that it answers the
same way as the chapter: embed the question, retrieve from RDS pgvector, and
generate a cited answer with Bedrock Converse.
"""

import json
import os

import boto3
import psycopg2

from corpus_data import chunk_text, documents

# -------------------------------------------------------------------------------
# Clients and configuration
# -------------------------------------------------------------------------------
SECRETS = boto3.client("secretsmanager")
BEDROCK = boto3.client("bedrock-runtime")

TEXT_MODEL = os.environ["TEXT_MODEL"]
EMBED_MODEL = os.environ["EMBED_MODEL"]

# The guardrail is optional so the stack deploys before one exists. Create it
# with `uv run underwriting-agent/guardrails.py create` in the chapter, then pass
# the id and version as stack parameters. With both set, the guardrail runs
# inside the Converse call, the same way the chapter's ask path applies it.
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "")

BLOCKED_OUTPUT = "The assistant cannot provide that answer under the credit guardrail."

# Same system prompt as the chapter's ask path (ASK_SYSTEM in agent.py), so a
# question answered here reads the same as one answered locally.
SYSTEM = (
    "You are a credit underwriting assistant for a lender to the solar sector."
    " Answer the question using only the context passages provided. If the answer"
    " is not in the context, say you cannot find it in the knowledge base. After"
    " each claim, cite the source document id in square brackets. Be concise and"
    " precise. Write in plain prose with no em dashes; use commas instead."
)


# -------------------------------------------------------------------------------
# Database connection and embedding helpers
# -------------------------------------------------------------------------------
def _connect():
    secret = json.loads(
        SECRETS.get_secret_value(SecretId=os.environ["DB_SECRET_ARN"])["SecretString"]
    )
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=os.environ["PGPORT"],
        dbname=os.environ["PGDATABASE"],
        user=secret["username"],
        password=secret["password"],
    )


def _embed(text):
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    resp = BEDROCK.invoke_model(modelId=EMBED_MODEL, body=body)
    return json.loads(resp["body"].read())["embedding"]


# -------------------------------------------------------------------------------
# Vector store setup and search
# -------------------------------------------------------------------------------
def _ensure_store(conn):
    cur = conn.cursor()
    cur.execute("create extension if not exists vector")
    cur.execute(
        "create table if not exists kb_chunks ("
        " id bigserial primary key, doc_id text, chunk_index int,"
        " content text, embedding vector(1024))"
    )
    cur.execute(
        "create index if not exists kb_chunks_embedding_idx"
        " on kb_chunks using hnsw (embedding vector_cosine_ops)"
    )
    cur.execute("select count(*) from kb_chunks")
    if cur.fetchone()[0] == 0:
        for doc_id, text in documents():
            for i, chunk in enumerate(chunk_text(text)):
                literal = "[" + ",".join(str(x) for x in _embed(chunk)) + "]"
                cur.execute(
                    "insert into kb_chunks (doc_id, chunk_index, content, embedding)"
                    " values (%s, %s, %s, %s::vector)",
                    (doc_id, i, chunk, literal),
                )
    conn.commit()
    cur.close()


def _search(conn, query, k=5):
    literal = "[" + ",".join(str(x) for x in _embed(query)) + "]"
    cur = conn.cursor()
    cur.execute(
        "select doc_id, content from kb_chunks order by embedding <=> %s::vector limit %s",
        (literal, k),
    )
    rows = cur.fetchall()
    cur.close()
    return rows


# -------------------------------------------------------------------------------
# Lambda entry point
# -------------------------------------------------------------------------------
def answer(event, context):
    """Lambda entry point: retrieve, then generate a grounded, cited answer."""
    query = json.loads(event.get("body") or "{}").get("query", "")
    conn = _connect()
    _ensure_store(conn)
    hits = _search(conn, query)
    conn.close()

    passages = "\n\n".join(f"[{doc_id}] {content}" for doc_id, content in hits)
    kwargs = {
        "modelId": TEXT_MODEL,
        "system": [{"text": SYSTEM}],
        "inferenceConfig": {"maxTokens": 600, "temperature": 0.0},
    }
    if GUARDRAIL_ID and GUARDRAIL_VERSION:
        # the retrieved passages are the grounding source and the question is the
        # query, so the contextual grounding filter can score the answer
        content = [
            {
                "guardContent": {
                    "text": {"text": passages, "qualifiers": ["grounding_source"]}
                }
            },
            {"guardContent": {"text": {"text": query, "qualifiers": ["query"]}}},
        ]
        kwargs["guardrailConfig"] = {
            "guardrailIdentifier": GUARDRAIL_ID,
            "guardrailVersion": GUARDRAIL_VERSION,
        }
    else:
        content = [{"text": f"Question: {query}\n\n{passages}"}]
    kwargs["messages"] = [{"role": "user", "content": content}]

    resp = BEDROCK.converse(**kwargs)
    blocks = resp["output"]["message"]["content"]
    text = "".join(b.get("text", "") for b in blocks) if blocks else BLOCKED_OUTPUT
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"answer": text, "sources": [doc_id for doc_id, _ in hits]}),
    }
