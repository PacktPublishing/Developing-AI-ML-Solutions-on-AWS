"""Benchmark the two embedding stages on CPU against Apple GPU (MPS).

uv run --project . python bench_device.py [faces_dir] [-n N]
"""

import argparse
import statistics
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facenet_pytorch import MTCNN, InceptionResnetV1


def sync(device: str) -> None:
    """Wait for queued GPU work, so a timing covers it rather than the enqueue."""
    if device == "mps":
        torch.mps.synchronize()


def detect_all(images: list[Image.Image]) -> tuple[torch.Tensor, float]:
    """Detect and align every face on CPU, returning the batch and the elapsed seconds."""
    detector = MTCNN(keep_all=True, device=torch.device("cpu"))
    t0 = time.perf_counter()
    faces = []
    for img in images:
        boxes, _ = detector.detect(img)
        if boxes is None:
            continue
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        extracted = detector.extract(img, boxes, save_path=None)
        faces.append(extracted[int(np.argmax(areas))])
    return torch.stack(faces), time.perf_counter() - t0


def bench_embed(faces: torch.Tensor, device: str) -> dict:
    """Time model load, one-at-a-time embedding, and one batched embedding."""
    t0 = time.perf_counter()
    model = InceptionResnetV1(pretrained="vggface2").eval().to(torch.device(device))
    sync(device)
    load = time.perf_counter() - t0

    # Warm up BOTH shapes: MPS compiles kernels per shape, so a cold batch would
    # otherwise be timed as compilation rather than as work.
    with torch.no_grad():
        model(faces[:1].to(torch.device(device)))
        sync(device)
        model(faces.to(torch.device(device)))
        sync(device)

    per_image = []
    with torch.no_grad():
        for i in range(len(faces)):
            t0 = time.perf_counter()
            model(faces[i : i + 1].to(torch.device(device)))
            sync(device)
            per_image.append(time.perf_counter() - t0)

    batch_runs = []
    with torch.no_grad():
        for _ in range(5):
            t0 = time.perf_counter()
            batch = model(faces.to(torch.device(device)))
            sync(device)
            batch_runs.append(time.perf_counter() - t0)
    batched = statistics.median(batch_runs)

    return {
        "load": load,
        "median": statistics.median(per_image),
        "loop_total": sum(per_image),
        "batched": batched,
        "embeddings": batch.cpu().numpy(),
    }


def main() -> None:
    """Compare CPU and MPS on the chapter's registered ID photos."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("faces", nargs="?", default="../data/generated/faces")
    ap.add_argument("-n", type=int, default=40)
    args = ap.parse_args()

    paths = sorted(Path(args.faces).glob("registered/*/id.jpg"))[: args.n]
    images = [Image.open(BytesIO(p.read_bytes())).convert("RGB") for p in paths]
    mps = torch.backends.mps.is_available()
    print(f"torch {torch.__version__}, mps available {mps}")
    print(f"{len(images)} registered ID photos")

    faces, detect_secs = detect_all(images)
    print("\ndetect + align (CPU only, MTCNN cannot run on MPS)")
    print(f"total: {detect_secs:.2f} s, {detect_secs / len(faces) * 1000:.1f} ms/face")

    results = {}
    for device in ("cpu", "mps"):
        if device == "mps" and not mps:
            continue
        r = bench_embed(faces, device)
        results[device] = r
        print(f"\nembed on {device}")
        print(f"model load: {r['load']:.2f} s")
        print(
            f"one at a time: {r['median'] * 1000:.1f} ms/face, "
            f"{r['loop_total']:.2f} s total"
        )
        print(
            f"one batch of {len(faces)}: {r['batched'] * 1000:.1f} ms, "
            f"{r['batched'] / len(faces) * 1000:.1f} ms/face"
        )

    if len(results) == 2:
        cpu, gpu = results["cpu"], results["mps"]
        a, b = cpu["embeddings"], gpu["embeddings"]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        print(
            f"\nagreement: min cosine {cos.min():.6f}, "
            f"max abs diff {np.abs(a - b).max():.2e}"
        )
        print(
            f"speedup: {cpu['median'] / gpu['median']:.2f}x one at a time, "
            f"{cpu['batched'] / gpu['batched']:.2f}x batched"
        )
        run_cpu = detect_secs + cpu["loop_total"]
        run_mps = detect_secs + gpu["loop_total"]
        print(
            f"whole registration run (detect + embed): cpu {run_cpu:.2f} s, "
            f"cpu detect + mps embed {run_mps:.2f} s, {run_cpu / run_mps:.2f}x"
        )


if __name__ == "__main__":
    main()
