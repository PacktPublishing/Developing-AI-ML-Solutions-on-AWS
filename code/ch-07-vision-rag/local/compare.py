"""Compare two face photos: cosine similarity + a 4-panel Captum explanation.

    uv run python compare.py a.jpg b.jpg

Prints the cosine similarity and saves /tmp/ch07_compare.png — four panels on a
dark canvas: Image A | Attribution A | Image B | Attribution B, each face beside
its Integrated Gradients heatmap (inferno, both scaled to a shared max).
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_embedder import FaceEmbedder  # noqa: E402
from face_explainer import explain, heat_overlay  # noqa: E402

CELL = 256


def _up(rgb):
    return Image.fromarray(rgb).resize((CELL, CELL), Image.LANCZOS)


def _font(size, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    """Compare two local images and print the decision."""
    a, b = sys.argv[1], sys.argv[2]
    emb = FaceEmbedder(device="cpu")
    out = explain(emb, Path(a).read_bytes(), Path(b).read_bytes())
    if out is None:
        print("no face detected in one of the images")
        return
    sim = out["similarity"]
    print(f"cosine similarity: {sim:+.4f}")

    vmax = max(float(out["sal_a"].max()), float(out["sal_b"].max()))
    panels = [
        (_up(out["a"]), "Image A"),
        (_up(heat_overlay(out["a"], out["sal_a"], vmax)), "Attribution A"),
        (_up(out["b"]), "Image B"),
        (_up(heat_overlay(out["b"], out["sal_b"], vmax)), "Attribution B"),
    ]

    pad, gap, top, banner = 20, 14, 34, 52
    w = pad * 2 + CELL * 4 + gap * 3
    h = top + CELL + banner
    canvas = Image.new("RGB", (w, h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    tf, sf = _font(15), _font(22, bold=True)

    x = pad
    for img, title in panels:
        canvas.paste(img, (x, top))
        tw = draw.textbbox((0, 0), title, font=tf)[2]
        draw.text(
            (x + (CELL - tw) // 2, top - 24), title, fill=(230, 230, 230), font=tf
        )
        x += CELL + gap

    label = f"Similarity Score: {sim:+.4f}"
    tw = draw.textbbox((0, 0), label, font=sf)[2]
    draw.text(
        ((w - tw) // 2, top + CELL + (banner - 24) // 2),
        label,
        fill=(255, 255, 255),
        font=sf,
    )
    canvas.save("/tmp/ch07_compare.png")
    print("wrote /tmp/ch07_compare.png")


if __name__ == "__main__":
    main()
