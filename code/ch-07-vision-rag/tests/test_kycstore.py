"""The vector store: concurrent schema creation, and search that answers correctly.

The concurrency test is a regression guard. Eight identity photos uploaded together
fire eight registrations at once, and against a cold database they all reach
ensure_schema simultaneously. Postgres `IF NOT EXISTS` is not concurrency-safe, so
before the advisory lock three of the eight died with a UniqueViolation on
pg_extension_name_index and the applicant never got registered.
"""

import random
from concurrent.futures import ThreadPoolExecutor

import kycstore

DIM = kycstore.EMBEDDING_DIM


def _vector(seed: int) -> list[float]:
    """Return a deterministic unit vector, standing in for a face embedding."""
    rng = random.Random(seed)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


class _Array(list):
    """A list that answers tolist(), which is all kycstore needs of a vector."""

    def tolist(self):
        """Return the values as a plain list."""
        return list(self)


def _fresh_schema() -> None:
    """Drop the table and extension so the next call creates them from cold."""
    with kycstore.connect() as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS faces")
        cur.execute("DROP EXTENSION IF EXISTS vector CASCADE")


def test_concurrent_registrations_do_not_race_on_the_schema():
    """Eight simultaneous first-callers must all get a schema, not a UniqueViolation."""
    _fresh_schema()

    def register(i: int) -> None:
        with kycstore.connect() as conn:
            kycstore.ensure_schema(conn)
            kycstore.insert(
                conn, f"subject_{i:03d}", f"registered/s{i}/id.jpg", _Array(_vector(i))
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        # raises here if any worker failed, which is exactly the old defect
        list(pool.map(register, range(8)))

    with kycstore.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM faces")
        assert cur.fetchone()[0] == 8


def test_the_same_photo_registers_once():
    """A redelivered S3 event must not duplicate an identity."""
    _fresh_schema()
    with kycstore.connect() as conn:
        kycstore.ensure_schema(conn)
        for _ in range(3):
            kycstore.insert(
                conn, "subject_000", "registered/s0/id.jpg", _Array(_vector(0))
            )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM faces WHERE s3_key = 'registered/s0/id.jpg'"
            )
            assert cur.fetchone()[0] == 1


def test_search_returns_the_nearest_subject_first():
    """A probe close to one registered face ranks that subject above the others."""
    _fresh_schema()
    with kycstore.connect() as conn:
        kycstore.ensure_schema(conn)
        for i in range(5):
            kycstore.insert(
                conn, f"subject_{i:03d}", f"registered/s{i}/id.jpg", _Array(_vector(i))
            )

        # a probe one small step away from subject_002's vector
        target = _vector(2)
        probe = _Array([x + 0.01 for x in target])
        matches = kycstore.search(conn, probe, k=5)

    assert matches[0]["subject"] == "subject_002"
    assert matches[0]["score"] > 0.9
    assert [m["score"] for m in matches] == sorted(
        (m["score"] for m in matches), reverse=True
    )


def test_search_can_exclude_the_probe_itself():
    """A 1:N search must not rank an object against its own stored row."""
    _fresh_schema()
    with kycstore.connect() as conn:
        kycstore.ensure_schema(conn)
        for i in range(3):
            kycstore.insert(
                conn, f"subject_{i:03d}", f"registered/s{i}/id.jpg", _Array(_vector(i))
            )

        own = _Array(_vector(1))
        assert kycstore.search(conn, own, k=3)[0]["subject"] == "subject_001"
        excluded = kycstore.search(conn, own, k=3, exclude_key="registered/s1/id.jpg")

    assert all(m["subject"] != "subject_001" for m in excluded)
