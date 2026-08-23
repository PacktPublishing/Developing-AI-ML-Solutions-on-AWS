"""Provision a throwaway pgvector for the store tests, as Chapters 3 and 9 do.

Container start-up and the environment kycstore reads both belong here rather than
at the top of a test module, so the test files keep ordinary top-level imports.
"""

import atexit
import os
import socket
import sys
import time
from pathlib import Path

import pytest

PG_PORT = 5508  # distinct from the local stack on 5507, so `make up` can stay running

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _pg_up() -> bool:
    """Return True if something already answers on the test port."""
    try:
        socket.create_connection(("localhost", PG_PORT), timeout=2).close()
        return True
    except OSError:
        return False


def _start_pg() -> None:
    """Run pgvector in Docker and wait for it to accept connections."""
    import docker

    try:
        engine = docker.from_env()
    except Exception:
        pytest.skip("no Docker daemon for pgvector", allow_module_level=True)
    container = engine.containers.run(
        "pgvector/pgvector:pg16",
        environment={
            "POSTGRES_USER": "kyc",
            "POSTGRES_PASSWORD": "kyc",
            "POSTGRES_DB": "kyc",
        },
        ports={"5432/tcp": PG_PORT},
        detach=True,
        remove=True,
    )
    atexit.register(container.stop)

    import psycopg

    for _ in range(60):
        try:
            psycopg.connect(
                host="localhost", port=PG_PORT, user="kyc", password="kyc", dbname="kyc"
            ).close()
            return
        except Exception:
            time.sleep(1)
    pytest.skip("pgvector did not become ready", allow_module_level=True)


if not _pg_up():
    _start_pg()

os.environ.update(
    PGHOST="localhost",
    PGPORT=str(PG_PORT),
    PGUSER="kyc",
    PGPASSWORD="kyc",
    PGDATABASE="kyc",
)
os.environ.pop("DB_IAM_AUTH", None)
os.environ.pop("KYC_DSN", None)
