"""Compare Lambda: cosine similarity between two face photos.

    POST /compare  {"a": "img1.jpg", "b": "img2.jpg"}
    ->  {"a", "b", "similarity"}          # 1.0 = identical, ~0 = unrelated

With {"explain": true} it also runs Captum Integrated Gradients (both faces),
writes a 2-panel purple saliency grid to S3, and returns its key. Explanation is
opt-in because it is much heavier than a plain embed-and-compare.
"""

import json
import logging
import os
import uuid
from io import BytesIO

import boto3
import numpy as np

os.chdir("/tmp")
log = logging.getLogger()
log.setLevel(logging.INFO)

DEFAULT_BUCKET = os.environ.get("IMAGES_S3_BUCKET", "")

from face_embedder import FaceEmbedder

_embedder = FaceEmbedder(device="cpu")


def _explain_png(out) -> bytes:
    """4 panels: each face beside its Integrated Gradients heatmap (shared scale)."""
    from face_explainer import heat_overlay
    from PIL import Image, ImageDraw, ImageFont

    vmax = max(float(out["sal_a"].max()), float(out["sal_b"].max()))
    cell = 256

    def up(rgb):
        return Image.fromarray(rgb).resize((cell, cell), Image.LANCZOS)

    panels = [
        (up(out["a"]), "Image A"),
        (up(heat_overlay(out["a"], out["sal_a"], vmax)), "Attribution A"),
        (up(out["b"]), "Image B"),
        (up(heat_overlay(out["b"], out["sal_b"], vmax)), "Attribution B"),
    ]
    pad, gap, top, banner = 20, 14, 30, 48
    w = pad * 2 + cell * 4 + gap * 3
    canvas = Image.new("RGB", (w, top + cell + banner), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x = pad
    for img, title in panels:
        canvas.paste(img, (x, top))
        draw.text((x + 4, top - 16), title, fill=(230, 230, 230), font=font)
        x += cell + gap
    draw.text(
        (pad, top + cell + 14),
        f"Similarity Score: {out['similarity']:+.4f}",
        fill=(255, 255, 255),
        font=font,
    )
    buf = BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()


def lambda_handler(event, context):
    """Compare two face images and return the match decision."""
    body = event.get("body", event)
    p = json.loads(body) if isinstance(body, str) else body
    a, b = p.get("a"), p.get("b")
    bucket = p.get("bucket") or DEFAULT_BUCKET
    if not a or not b:
        return {"statusCode": 400, "body": json.dumps({"error": "a and b required"})}

    s3 = boto3.client("s3")
    a_bytes = s3.get_object(Bucket=bucket, Key=a)["Body"].read()
    b_bytes = s3.get_object(Bucket=bucket, Key=b)["Body"].read()

    ea, eb = _embedder.get_embedding(a_bytes), _embedder.get_embedding(b_bytes)
    if ea is None or eb is None:
        return {
            "statusCode": 200,
            "body": json.dumps({"a": a, "b": b, "similarity": None}),
        }

    sim = float(np.dot(ea, eb))  # embeddings are L2-normalized -> dot = cosine
    result = {"a": a, "b": b, "similarity": round(sim, 4)}

    if p.get("explain"):
        from face_explainer import explain

        out = explain(_embedder, a_bytes, b_bytes)
        if out is not None:
            key = f"explanations/{uuid.uuid4().hex}.png"
            s3.put_object(
                Bucket=bucket, Key=key, Body=_explain_png(out), ContentType="image/png"
            )
            result["explanation_key"] = key

    log.info("%s vs %s -> %.3f", a, b, sim)
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result),
    }
