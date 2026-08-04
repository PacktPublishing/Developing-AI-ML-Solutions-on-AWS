# /// script
# requires-python = ">=3.12"
# dependencies = ["kaggle", "pillow"]
# ///
"""Build the KYC face pairs from a real, permissively-licensed face set.

Real KYC selfie/ID pairs are confidential, and every public "selfie + ID" set on
Kaggle is NonCommercial/NoDerivatives — unusable in a commercial book. So the
enrolled "ID photo" is a real face from **UTKFace** (CC0 public domain: real,
diverse, non-celebrity, one image per person), and the matching "selfie" is a
*simulated re-capture of the same face* — a light, identity-preserving
augmentation (small rotation, lighting, crop, JPEG re-encode). Same person, a
second capture: exactly the 1:1 check the chapter demonstrates. No real
identities, no license or privacy exposure. Images are fetched to a gitignored
folder and never committed.

The optional `--hf` path swaps the classic augmentation for a Hugging Face
img2img model (heavier, nondeterministic, and it can *drift the identity*, which
silently turns a match into a non-match — off by default for that reason).

Output (under a gitignored dir):
    <out>/enrolled/<subject>/id.jpg      the real face (enrolled)
    <out>/probe/<subject>/selfie.jpg     the simulated re-capture (the match probe)
    <out>/impostors/<subject>/selfie.jpg  a different person's selfie (non-match)
    <out>/pairs.csv                      subject,id_path,selfie_path,impostor_path

Run (Kaggle token in the repo .env as KAGGLE_API_KEY):
    uv run etl/build_pairs.py --n 40
"""

import argparse
import csv
import io
import os
import random
from pathlib import Path

from PIL import Image, ImageEnhance

DATASET = "samuelagyemang/utkface"  # CC0 public domain


def ensure_kaggle_token() -> None:
    """Map the repo's KAGGLE_API_KEY onto the name the Kaggle client reads."""
    if not os.environ.get("KAGGLE_API_TOKEN") and os.environ.get("KAGGLE_API_KEY"):
        os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_API_KEY"]


def fetch_faces(cache: Path) -> list[Path]:
    """Download + unzip UTKFace once into the gitignored cache; return image paths."""
    images = sorted(cache.rglob("*.jpg"))
    if not images:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        cache.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(DATASET, path=str(cache), unzip=True, quiet=False)
        images = sorted(cache.rglob("*.jpg"))
    if not images:
        raise SystemExit(f"no images found under {cache}")
    return images


def id_crop(path: Path, size: int = 256) -> Image.Image:
    """Crop the real face to a square enrolled ID photo at a fixed size."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    side = min(w, h)
    im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    return im.resize((size, size), Image.LANCZOS)


def make_selfie(im: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate a second capture of the SAME face with identity-preserving jitter.

    Small rotation + lighting/colour + a slight zoom/crop, then a JPEG round-trip
    so it is a genuinely different file (a re-photograph), not a copy — while the
    face stays the same person, which the embedder should still match.
    """
    out = im.rotate(
        rng.uniform(-12, 12), resample=Image.BICUBIC, fillcolor=(127, 127, 127)
    )
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.85, 1.15))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.9, 1.15))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.9, 1.1))
    w, h = out.size
    z = rng.uniform(0.02, 0.12)
    out = out.crop(
        (int(w * z / 2), int(h * z / 2), int(w * (1 - z / 2)), int(h * (1 - z / 2)))
    ).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_selfie_hf(im: Image.Image, strength: float) -> Image.Image:
    """Re-capture the face with a Hugging Face img2img model (can drift identity)."""
    import torch
    from diffusers import AutoPipelineForImage2Image

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float32
    )
    result = pipe(
        prompt="a passport-style photo of the same person, neutral lighting",
        image=im,
        strength=strength,
        guidance_scale=0.0,
        num_inference_steps=2,
    ).images[0]
    return result.resize(im.size, Image.LANCZOS)


def main() -> None:
    """Fetch real faces and emit enrolled/probe/impostor triples."""
    ap = argparse.ArgumentParser(description="Build KYC face pairs from UTKFace (CC0).")
    ap.add_argument("--n", type=int, default=40, help="how many subjects")
    ap.add_argument("--out", type=Path, default=Path("data/generated/faces"))
    ap.add_argument("--cache", type=Path, default=Path("data/generated/utkface"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hf", action="store_true", help="use HF img2img (drift risk)")
    ap.add_argument("--strength", type=float, default=0.3, help="HF img2img strength")
    args = ap.parse_args()

    ensure_kaggle_token()
    rng = random.Random(args.seed)
    images = fetch_faces(args.cache)
    subjects = rng.sample(images, min(args.n, len(images)))

    rows = []
    for i, src in enumerate(subjects):
        subject = f"subject_{i:03d}"
        ident = id_crop(src)
        selfie = make_selfie_hf(ident, args.strength) if args.hf else make_selfie(ident, rng)
        # a non-match "selfie" from a different subject (wraps around the list)
        impostor_src = subjects[(i + 1) % len(subjects)]
        impostor = make_selfie(id_crop(impostor_src), rng)

        id_path = args.out / "enrolled" / subject / "id.jpg"
        selfie_path = args.out / "probe" / subject / "selfie.jpg"
        impostor_path = args.out / "impostors" / subject / "selfie.jpg"
        for p, img in (
            (id_path, ident),
            (selfie_path, selfie),
            (impostor_path, impostor),
        ):
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p, format="JPEG", quality=92)
        rows.append((subject, id_path, selfie_path, impostor_path))

    manifest = args.out / "pairs.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "id_path", "selfie_path", "impostor_path"])
        w.writerows(rows)
    print(f"{len(rows)} subjects -> {args.out} (manifest {manifest})")


if __name__ == "__main__":
    main()
