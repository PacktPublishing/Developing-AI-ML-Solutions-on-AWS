"""pgvector round-trip: embed a small synthetic corpus and search it, no cloud."""

import atexit
import os
import socket
import time

import pytest

PG_PORT = 5546


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

import embed_memos  # noqa: E402
import gen_memos  # noqa: E402
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
def test_exact_chunk_ranks_its_own_memo_first(monkeypatch, tmp_path):
    """An exact memo chunk ranks its own source loan first."""
    monkeypatch.setattr(embed_memos, "embed", _fake_embed)
    monkeypatch.setattr(embed_memos, "get_runtime", lambda: None)
    monkeypatch.setattr(stores, "embed", _fake_embed)

    memo_dir = tmp_path / "memos"
    gen_memos.generate(memo_dir, count=8, seed=7, messy=False)
    embed_memos.seed(memo_dir)

    # a structured memo has a unique first chunk (its business name and address)
    structured = next(
        p
        for p in sorted(memo_dir.glob("*.txt"))
        if embed_memos.parse_memo(p)[2].startswith("SME Credit Memo")
    )
    loan_id, _, body = embed_memos.parse_memo(structured)
    chunk = embed_memos.chunk_text(body)[0]

    hits = stores.search(None, chunk, k=3)

    assert len(hits) == 3
    assert hits[0][0] == loan_id
    assert hits[0][3] == pytest.approx(1.0, abs=1e-6)
