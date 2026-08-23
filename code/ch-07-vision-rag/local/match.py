"""Match a probe face against the enrolled store (1:N), the local search Lambda.

uv run match.py <probe.jpg> [-k N] [--claim SUBJECT]

Without --claim it is a pure 1:N search (is this face anyone we know?). With
--claim it is KYC verification: matched only when the top hit IS the claimed
subject above the threshold, so an impostor presenting as one subject but
resolving to another is rejected.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_embedder import FaceEmbedder, best_devices
from kycstore import connect, search

MATCH = 0.70


def main() -> None:
    """Search the enrolled faces for the closest matches to a probe image."""
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--claim", default=None, help="the subject the probe claims to be")
    args = ap.parse_args()

    device, detector_device = best_devices()
    emb = FaceEmbedder(device=device, detector_device=detector_device).get_embedding(
        Path(args.probe).read_bytes()
    )
    if emb is None:
        print("no face detected in the probe")
        return

    with connect() as conn:
        matches = [(m["subject"], m["score"]) for m in search(conn, emb, args.k)]

    top = matches[0] if matches else None
    if args.claim is None:
        matched = bool(top and top[1] >= MATCH)
        print(f"probe {args.probe}\n  matched={matched} (threshold {MATCH})")
    else:
        # KYC verification: the top hit must BE the claimed subject, above threshold.
        matched = bool(top and top[0] == args.claim and top[1] >= MATCH)
        identified = top[0] if top else None
        print(
            f"probe {args.probe}\n  claim={args.claim} identified={identified} "
            f"matched={matched} (threshold {MATCH})"
        )
    for subject, score in matches:
        print(f"  {subject}  {score:+.4f}")


if __name__ == "__main__":
    main()
