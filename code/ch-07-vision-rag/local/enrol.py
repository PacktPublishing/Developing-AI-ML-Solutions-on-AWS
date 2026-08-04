"""Enrol the ETL's ID photos into the local pgvector store.

The local mirror of the enrolment Lambda: same `faces` table + index-first HNSW,
but it reads `enrolled/{subject}/id.jpg` from the filesystem (what
`etl/build_pairs.py` wrote) instead of firing on an S3 upload. Same embedder,
same SQL — only the source and the DSN change.

    uv run enrol.py [faces_dir]     # default ../data/generated/faces
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_embedder import FaceEmbedder  # noqa: E402
from kycdb import connect, ensure_schema  # noqa: E402


def main() -> None:
    """Embed every enrolled ID photo into the faces table."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/generated/faces")
    embedder = FaceEmbedder(device="cpu")
    conn = connect()
    ensure_schema(conn)

    enrolled = 0
    for id_path in sorted(root.glob("enrolled/*/id.jpg")):
        subject = id_path.parent.name
        key = f"enrolled/{subject}/id.jpg"
        emb = embedder.get_embedding(id_path.read_bytes())
        if emb is None:
            print(f"no face in {key}")
            continue
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO faces (subject, s3_key, embedding) "
                "VALUES (%s, %s, %s::vector) ON CONFLICT (s3_key) DO NOTHING",
                (subject, key, str(emb.tolist())),
            )
        conn.commit()
        enrolled += 1
    print(f"enrolled {enrolled} face(s)")


if __name__ == "__main__":
    main()
