"""The vector store seam: local pgvector in Docker, Aurora Serverless on AWS.

The chapter's one definition of how to reach the faces table. Local runs read a plain
DSN; on AWS the express Aurora cluster authenticates through its internet gateway with
IAM only, so the password is a short-lived RDS auth token over TLS. DB_IAM_AUTH=1
selects that path, exactly as Chapter 3's store does.

Connections come from a pool rather than a module-level singleton. The serving
container runs four gunicorn threads and SageMaker sends up to four concurrent
invocations, so a single shared connection would be used from several threads at
once; it also means one failed statement can only spoil the connection it ran on,
which the pool then resets on return instead of poisoning every later request.
"""

import os

import psycopg
from psycopg_pool import ConnectionPool

EMBEDDING_DIM = 512
_pool: ConnectionPool | None = None


def _conninfo() -> tuple[str, dict]:
    """Return the connection string and keyword arguments for this environment."""
    dsn = os.environ.get("KYC_DSN")
    if dsn and os.environ.get("DB_IAM_AUTH") != "1":
        return dsn, {}

    host = os.environ.get("PGHOST", "localhost")
    port = int(os.environ.get("PGPORT", "5507"))
    user = os.environ.get("PGUSER", "kyc")
    dbname = os.environ.get("PGDATABASE", "kyc")
    kwargs = {"host": host, "port": port, "user": user, "dbname": dbname}
    if os.environ.get("DB_IAM_AUTH") == "1":
        kwargs["sslmode"] = "require"
    else:
        kwargs["password"] = os.environ.get("PGPASSWORD", "kyc")
    return "", kwargs


def _password() -> str | None:
    """Mint a fresh RDS auth token, which expires and must not be cached with the pool."""
    if os.environ.get("DB_IAM_AUTH") != "1":
        return None
    import boto3

    region = os.environ.get(
        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    host = os.environ.get("PGHOST", "localhost")
    port = int(os.environ.get("PGPORT", "5432"))
    user = os.environ.get("PGUSER", "postgres")
    return boto3.client("rds", region_name=region).generate_db_auth_token(
        host, port, user
    )


class _IamConnection(psycopg.Connection):
    """A connection that mints its own RDS auth token as it opens.

    The pool outlives any single token, so the credential cannot be fixed when the
    pool is built; it has to be produced per connection.
    """

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs):
        """Open one connection, supplying a fresh token when IAM auth is in use."""
        if password := _password():
            kwargs["password"] = password
        return super().connect(conninfo, **kwargs)


def _configure(conn) -> None:
    """Autocommit, so a SELECT never leaves a transaction open on a pooled connection."""
    conn.autocommit = True


def pool() -> ConnectionPool:
    """Return the process's connection pool, opening it on first use.

    The token is minted per connection rather than once for the pool, because an RDS
    auth token is short lived and the pool outlives it.
    """
    global _pool
    if _pool is None:
        dsn, kwargs = _conninfo()

        _pool = ConnectionPool(
            conninfo=dsn,
            kwargs=kwargs,
            connection_class=_IamConnection,
            min_size=1,
            max_size=int(os.environ.get("KYC_POOL_SIZE", "5")),
            # An express cluster pauses when idle: check a connection before handing
            # it out, and wait long enough for the cluster to resume.
            check=ConnectionPool.check_connection,
            timeout=90,
            configure=_configure,
            open=True,
        )
    return _pool


def connect():
    """Borrow a connection from the pool.

    Callers use this as a context manager; the pool resets and returns the connection
    on exit, including after an error.
    """
    return pool().connection()


# Any constant works; it only has to be the same number in every process that runs
# the DDL below. This one is "ch07" as digits.
_SCHEMA_LOCK = 70_723_407


def ensure_schema(conn) -> None:
    """Create the faces table and its HNSW index up front (index-first).

    Serialized behind an advisory lock, because Postgres `IF NOT EXISTS` is not
    concurrency-safe: it looks, then inserts into the catalog, and two sessions
    arriving together both pass the look. Eight registration uploads land as eight
    near-simultaneous invocations on a cold endpoint, so this is not a rare race.
    The loser of that race sees `duplicate key value violates unique constraint
    "pg_extension_name_index"` and the whole registration fails with a 500.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_LOCK,))
        try:
            _create(cur)
        finally:
            # a session lock, not a transaction one: the pool runs with autocommit,
            # so there is no transaction whose end would release it
            cur.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_LOCK,))


def _create(cur) -> None:
    """Create the extension, the table, and the index, one caller at a time."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS faces (
            id        BIGSERIAL PRIMARY KEY,
            subject   TEXT NOT NULL,
            s3_key    TEXT UNIQUE NOT NULL,
            embedding VECTOR({EMBEDDING_DIM})
        )
    """)
    # the HNSW graph exists before any rows, so each write maintains the index
    # incrementally rather than paying for a rebuild later.
    cur.execute(
        "CREATE INDEX IF NOT EXISTS faces_hnsw ON faces "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def insert(conn, subject: str, key: str, vector) -> None:
    """Store one embedding, doing nothing if that object is already registered."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (subject, s3_key, embedding) "
            "VALUES (%s, %s, %s::vector) ON CONFLICT (s3_key) DO NOTHING",
            (subject, key, str(vector.tolist())),
        )


def search(conn, vector, k: int = 5, exclude_key: str | None = None) -> list[dict]:
    """Return the k nearest registered faces by cosine similarity.

    HNSW picks the candidates approximately; the score column is exact cosine, so
    sorting these k rows by score is an exact re-rank of an approximate shortlist.
    """
    vec = str(vector.tolist())
    with conn.cursor() as cur:
        cur.execute(
            "SELECT subject, (1 - (embedding <=> %s::vector)) AS score "
            "FROM faces WHERE s3_key IS DISTINCT FROM %s "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (vec, exclude_key, vec, k),
        )
        rows = [
            {"subject": r[0], "score": round(float(r[1]), 4)} for r in cur.fetchall()
        ]
    rows.sort(key=lambda m: m["score"], reverse=True)
    return rows
