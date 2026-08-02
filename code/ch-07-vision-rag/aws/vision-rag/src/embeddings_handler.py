"""Enrolment Lambda: embed enrolled identity photos from S3 into pgvector.

Enrolled images live under `enrolled/{subject}/...` in the bucket; an upload
there triggers this function, which embeds the face and stores it in one `faces`
table. The table and its HNSW index are created up front (index-first), so every
async write lands on an index that already exists.
"""

import json
import logging
import os

import boto3

os.chdir("/tmp")
log = logging.getLogger()
log.setLevel(logging.INFO)

DB_HOST = os.environ["PGHOST"]
DB_NAME = os.environ.get("PGDATABASE", "kyc")
DB_SECRET_ARN = os.environ["DB_SECRET_ARN"]
DEFAULT_BUCKET = os.environ.get("IMAGES_S3_BUCKET", "")

from face_embedder import FaceEmbedder  # noqa: E402

_embedder = FaceEmbedder(device="cpu")
_conn = None


def _db():
    global _conn
    import psycopg

    if _conn is None or _conn.closed:
        creds = json.loads(
            boto3.client("secretsmanager").get_secret_value(SecretId=DB_SECRET_ARN)[
                "SecretString"
            ]
        )
        _conn = psycopg.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=creds["username"],
            password=creds["password"],
            sslmode="require",
            connect_timeout=10,
        )
        with _conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id        BIGSERIAL PRIMARY KEY,
                    subject   TEXT NOT NULL,
                    s3_key    TEXT UNIQUE NOT NULL,
                    embedding VECTOR(512)
                )
            """)
            # index-first: the HNSW graph exists before any rows are inserted.
            cur.execute(
                "CREATE INDEX IF NOT EXISTS faces_hnsw ON faces "
                "USING hnsw (embedding vector_cosine_ops)"
            )
        _conn.commit()
    return _conn


def _pairs(event) -> list[tuple[str, str]]:
    """Return (bucket, key) pairs from an S3 trigger or a direct {keys:[...]}."""
    if "Records" in event:
        return [
            (r["s3"]["bucket"]["name"], r["s3"]["object"]["key"])
            for r in event["Records"]
        ]
    bucket = event.get("bucket") or DEFAULT_BUCKET
    return [(bucket, k) for k in event.get("keys", [])]


def lambda_handler(event, context):
    """Embed enrolment images from S3 into pgvector."""
    conn = _db()
    s3 = boto3.client("s3")
    enrolled = 0
    for bucket, key in _pairs(event):
        parts = key.split("/")  # enrolled/{subject}/{file}
        subject = parts[1] if len(parts) > 2 else parts[0]
        emb = _embedder.get_embedding(
            s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        )
        if emb is None:
            log.info("no face in %s", key)
            continue
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO faces (subject, s3_key, embedding) "
                "VALUES (%s, %s, %s::vector) ON CONFLICT (s3_key) DO NOTHING",
                (subject, key, str(emb.tolist())),
            )
        conn.commit()
        enrolled += 1
    log.info("enrolled %d face(s)", enrolled)
    return {"enrolled": enrolled}
