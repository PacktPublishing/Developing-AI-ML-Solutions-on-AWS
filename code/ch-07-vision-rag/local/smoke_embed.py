"""Smoke test for the face embedder.

Tier 1 (always): load facenet-pytorch on CPU, auto-download the VGGFace2
weights, and prove a 512-dim forward pass works. This is what settles the
"does facenet-pytorch install and run here" question.

Tier 2 (optional): if a face image path is given, run the full
detect -> align -> embed path end to end.

Run:
    uv run python smoke_embed.py [optional_face.jpg]

"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from face_embedder import EMBEDDING_DIM, FaceEmbedder  # noqa: E402


def main() -> None:
    """Embed one image and report the timing."""
    t0 = time.perf_counter()
    embedder = FaceEmbedder(device="cpu")
    print(
        f"[ok] model loaded (weights auto-downloaded if first run) "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    # Tier 1 — direct forward pass on a dummy aligned-face tensor.
    dummy = torch.rand(1, 3, 160, 160)
    with torch.no_grad():
        out = embedder.model(dummy)
    vec = out.squeeze(0).numpy()
    assert vec.shape == (EMBEDDING_DIM,), f"expected {EMBEDDING_DIM}, got {vec.shape}"
    assert np.isfinite(vec).all(), "non-finite values in embedding"
    print(
        f"[ok] forward pass: shape={vec.shape} dtype={vec.dtype} "
        f"norm={np.linalg.norm(vec):.3f}"
    )

    # Tier 2 — end-to-end on a real face, if one is provided.
    if len(sys.argv) > 1:
        img_bytes = Path(sys.argv[1]).read_bytes()
        t1 = time.perf_counter()
        emb = embedder.get_embedding(img_bytes)
        if emb is None:
            print(f"[--] no face detected in {sys.argv[1]}")
        else:
            print(
                f"[ok] end-to-end embed: shape={emb.shape} "
                f"in {time.perf_counter() - t1:.2f}s norm={np.linalg.norm(emb):.3f}"
            )
    else:
        print(
            "[..] no face image given — skipping end-to-end detect+embed "
            "(comes with the synthetic-data generator)"
        )


if __name__ == "__main__":
    main()
