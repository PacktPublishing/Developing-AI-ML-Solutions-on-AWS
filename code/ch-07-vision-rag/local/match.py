"""Match a probe face against the enrolled store (1:N) — the local search Lambda.

Embeds the probe, KNN-searches the `faces` table by cosine distance (the HNSW
index returns candidates; the score is exact cosine), and calls it a match if the
top score clears the threshold. Same SQL as the AWS search handler.

    uv run match.py <probe.jpg> [k]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_embedder import FaceEmbedder
from kycdb import connect

MATCH = 0.70


def main() -> None:
    """Search the enrolled faces for the closest matches to a probe image."""
    if len(sys.argv) < 2:
        sys.exit("usage: uv run match.py <probe.jpg> [k]")
    probe = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    emb = FaceEmbedder(device="cpu").get_embedding(Path(probe).read_bytes())
    if emb is None:
        print("no face detected in the probe")
        return

    vec = str(emb.tolist())
    with connect().cursor() as cur:
        cur.execute(
            "SELECT subject, (1 - (embedding <=> %s::vector)) AS score "
            "FROM faces ORDER BY embedding <=> %s::vector LIMIT %s",
            (vec, vec, k),
        )
        matches = [(r[0], round(float(r[1]), 4)) for r in cur.fetchall()]

    matched = bool(matches and matches[0][1] >= MATCH)
    print(f"probe {probe}\n  matched={matched} (threshold {MATCH})")
    for subject, score in matches:
        print(f"  {subject}  {score:+.4f}")


if __name__ == "__main__":
    main()
