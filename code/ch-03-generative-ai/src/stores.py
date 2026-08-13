"""The vector store seam: pgvector locally, RDS for PostgreSQL on AWS."""

import os

import psycopg2

from models import embed


def connect():
    """Connect to the vector store from the standard PG environment variables.

    Locally the password is the plain one. On AWS the express Aurora cluster authenticates
    through its internet gateway with IAM only, so the password is a short-lived RDS auth
    token and the connection is TLS; DB_IAM_AUTH=1 selects that path.
    """
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5544")
    user = os.environ.get("PGUSER", "underwriter")
    dbname = os.environ.get("PGDATABASE", "underwriting")
    if os.environ.get("DB_IAM_AUTH") == "1":
        import boto3

        region = os.environ.get("AWS_REGION", "us-east-1")
        token = boto3.client("rds", region_name=region).generate_db_auth_token(
            host, int(port), user
        )
        return psycopg2.connect(
            host=host,
            port=port,
            user=user,
            dbname=dbname,
            password=token,
            sslmode="require",
        )
    return psycopg2.connect(
        host=host,
        port=port,
        user=user,
        dbname=dbname,
        password=os.environ.get("PGPASSWORD", "underwriter"),
    )


def search(runtime, query: str, k: int = 5) -> list[tuple[str, str, float]]:
    """Return the k nearest knowledge-base chunks as (doc_id, content, similarity)."""
    query_vector = embed(runtime, [query])[0]
    literal = "[" + ",".join(str(x) for x in query_vector) + "]"
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "select doc_id, content, 1 - (embedding <=> %s::vector) as similarity"
        " from kb_chunks order by embedding <=> %s::vector limit %s",
        (literal, literal, k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
