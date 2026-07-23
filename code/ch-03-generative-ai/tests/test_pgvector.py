"""pgvector round-trip: seed the knowledge base and search it, no cloud.

Starts pgvector with docker-py (skips if there is no Docker daemon), the same
way the chapter's data-engineering tests provision DynamoDB Local. Embeddings
are stubbed with a deterministic function, so the test needs no Bedrock or
Ollama; it exercises the real plumbing, the extension, the schema, the vector
round-trip, and the cosine ordering.
"""

import atexit
import os
import socket
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "knowledge-base"))

PG_PORT = 5455


# -------------------------------------------------------------------------------
# Postgres container helpers
# -------------------------------------------------------------------------------
def _pg_up() -> bool:
    try:
        socket.create_connection(("localhost", PG_PORT), timeout=2).close()
        return True
    except OSError:
        return False


def _start_pg() -> None:
    import docker

    try:
        engine = docker.from_env()
    except Exception:
        pytest.skip("no Docker daemon for pgvector", allow_module_level=True)
    container = engine.containers.run(
        "pgvector/pgvector:pg16",
        environment={
            "POSTGRES_USER": "underwriter",
            "POSTGRES_PASSWORD": "underwriter",
            "POSTGRES_DB": "underwriting",
        },
        ports={"5432/tcp": PG_PORT},
        detach=True,
        remove=True,
    )
    atexit.register(container.stop)

    import psycopg2

    for _ in range(60):
        try:
            psycopg2.connect(
                host="localhost",
                port=PG_PORT,
                user="underwriter",
                password="underwriter",
                dbname="underwriting",
            ).close()
            return
        except Exception:
            time.sleep(1)
    pytest.skip("pgvector did not become ready", allow_module_level=True)


# -------------------------------------------------------------------------------
# Module-level provisioning
# -------------------------------------------------------------------------------
if not _pg_up():
    _start_pg()

os.environ.update(
    PGHOST="localhost",
    PGPORT=str(PG_PORT),
    PGUSER="underwriter",
    PGPASSWORD="underwriter",
    PGDATABASE="underwriting",
)

import corpus  # noqa: E402
import models  # noqa: E402
import stores  # noqa: E402


# -------------------------------------------------------------------------------
# Deterministic embedding stub
# -------------------------------------------------------------------------------
def _fake_embed(runtime, texts):
    """Deterministic embedding: same text maps to the same unit vector."""
    vectors = []
    for text in texts:
        vector = [0.0] * models.EMBED_DIM
        for i, code in enumerate(text.encode()):
            vector[i % models.EMBED_DIM] += code
        norm = sum(x * x for x in vector) ** 0.5 or 1.0
        vectors.append([x / norm for x in vector])
    return vectors


# -------------------------------------------------------------------------------
# Vector round-trip test
# -------------------------------------------------------------------------------
def test_seed_then_exact_chunk_ranks_first(monkeypatch):
    monkeypatch.setattr(corpus, "embed", _fake_embed)
    monkeypatch.setattr(corpus, "get_runtime", lambda: None)
    monkeypatch.setattr(stores, "embed", _fake_embed)

    corpus.seed()

    doc_id, text = corpus.documents()[0]
    chunk = corpus.chunk_text(text)[0]
    hits = stores.search(None, chunk, k=3)

    assert len(hits) == 3
    # an exact chunk embeds to an identical vector, so its own document ranks first
    assert hits[0][0] == doc_id
    assert hits[0][2] == pytest.approx(1.0, abs=1e-6)
