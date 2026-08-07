"""Search Lambda: embed a probe face and return the closest enrolled subjects.

    POST /match  {"key": "probes/applicant.jpg", "claim": "subject_000"}
    ->  {"key", "matches": [{"subject", "score"}, ...], "identified", "matched": bool}

pgvector's HNSW index returns the candidates; the score column is exact cosine,
so re-sorting by score is an exact re-rank of the top-k. Without `claim` it is a
1:N search and `matched` is the top score against a threshold; with `claim` it is
KYC verification and `matched` requires the top hit to BE the claimed subject.
"""

import json
import logging
import os

import boto3

from face_embedder import FaceEmbedder

os.chdir("/tmp")
log = logging.getLogger()
log.setLevel(logging.INFO)

DB_HOST = os.environ["PGHOST"]
DB_NAME = os.environ.get("PGDATABASE", "kyc")
DB_SECRET_ARN = os.environ["DB_SECRET_ARN"]
DEFAULT_BUCKET = os.environ.get("IMAGES_S3_BUCKET", "")
MATCH = 0.70

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
    return _conn


def lambda_handler(event, context):
    """Search enrolled faces for the closest matches to a probe image."""
    body = event.get("body", event)
    params = json.loads(body) if isinstance(body, str) else body
    key = params.get("key")
    bucket = params.get("bucket") or DEFAULT_BUCKET
    k = int(params.get("k", 5))
    claim = params.get("claim")  # optional: the subject the probe claims to be
    if not key:
        return {"statusCode": 400, "body": json.dumps({"error": "key required"})}

    image = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    emb = _embedder.get_embedding(image)
    if emb is None:
        return {
            "statusCode": 200,
            "body": json.dumps({"key": key, "matches": [], "matched": False}),
        }

    vec = str(emb.tolist())
    with _db().cursor() as cur:
        cur.execute(
            "SELECT subject, (1 - (embedding <=> %s::vector)) AS score "
            "FROM faces WHERE s3_key <> %s "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (vec, key, vec, k),
        )
        matches = [
            {"subject": r[0], "score": round(float(r[1]), 4)} for r in cur.fetchall()
        ]
    matches.sort(key=lambda m: m["score"], reverse=True)
    top = matches[0] if matches else None
    identified = top["subject"] if top else None
    # 1:N when no claim; KYC verification (top hit must be the claim) when claimed.
    matched = bool(
        top and top["score"] >= MATCH and (claim is None or identified == claim)
    )

    result = {
        "key": key,
        "matches": matches,
        "identified": identified,
        "matched": matched,
    }
    log.info("%s -> identified=%s matched=%s", key, identified, matched)
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result),
    }
