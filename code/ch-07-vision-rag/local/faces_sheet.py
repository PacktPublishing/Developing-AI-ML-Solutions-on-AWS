"""Contact sheet of the test set: enrolled ID, genuine probe, and impostor.

Saves /tmp/ch07_faces.png: one row per subject, three panels each, so the chapter
can show what "the same face twice" and "a different face under the same claim"
actually look like.

uv run --project . python faces_sheet.py [faces_dir] [-n N]
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL = 200
GAP = 10
PAD = 16
TOP = 30
COLUMNS = ("enrolled", "probe", "impostors")
HEADINGS = ("enrolled ID photo", "genuine probe selfie", "impostor selfie")


def _font(size: int):
    """Return Helvetica at this size, or PIL's bitmap font where it is absent."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def _cell(root: Path, column: str, subject: str) -> Image.Image:
    """Load one panel, square-cropped to the sheet's cell size."""
    name = "id.jpg" if column == "enrolled" else "selfie.jpg"
    im = Image.open(root / column / subject / name).convert("RGB")
    return im.resize((CELL, CELL), Image.Resampling.LANCZOS)


def main() -> None:
    """Build the contact sheet over the first N subjects."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("faces", nargs="?", default="../data/generated/faces")
    ap.add_argument("-n", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.faces)
    subjects = sorted(p.name for p in (root / "enrolled").iterdir() if p.is_dir())[
        : args.n
    ]
    if not subjects:
        raise SystemExit(f"no subjects under {root}/enrolled")

    width = PAD * 2 + CELL * 3 + GAP * 2
    height = TOP + PAD + len(subjects) * (CELL + GAP) - GAP + PAD
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for c, heading in enumerate(HEADINGS):
        x = PAD + c * (CELL + GAP)
        draw.text(
            (x + CELL // 2, TOP // 2),
            heading,
            fill=(20, 20, 20),
            font=_font(15),
            anchor="mm",
        )

    for r, subject in enumerate(subjects):
        y = TOP + PAD // 2 + r * (CELL + GAP)
        for c, column in enumerate(COLUMNS):
            sheet.paste(_cell(root, column, subject), (PAD + c * (CELL + GAP), y))

    out = Path("/tmp/ch07_faces.png")
    sheet.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
