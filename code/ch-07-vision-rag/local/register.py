"""Register the ETL's ID photos into the local pgvector store.

uv run register.py [faces_dir]     # default ../data/generated/faces
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_embedder import FaceEmbedder, best_devices
from kycstore import connect, ensure_schema, insert


def main() -> None:
    """Embed every registered ID photo into the faces table."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/generated/faces")
    device, detector_device = best_devices()
    embedder = FaceEmbedder(device=device, detector_device=detector_device)
    registered = 0
    with connect() as conn:
        ensure_schema(conn)
        for id_path in sorted(root.glob("registered/*/id.jpg")):
            subject = id_path.parent.name
            key = f"registered/{subject}/id.jpg"
            emb = embedder.get_embedding(id_path.read_bytes())
            if emb is None:
                print(f"no face in {key}")
                continue
            insert(conn, subject, key, emb)
            registered += 1
    print(f"registered {registered} face(s)")


if __name__ == "__main__":
    main()
