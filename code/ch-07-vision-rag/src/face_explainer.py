"""Explain a face comparison with Captum Integrated Gradients, both directions.

out = explain(embedder, a_bytes, b_bytes)
# out = {"similarity", "a", "sal_a", "b", "sal_b"}  (RGB uint8 + HxW [0,1])
"""

from io import BytesIO

import numpy as np
import torch
from captum.attr import IntegratedGradients
from PIL import Image


def _aligned(embedder, image_bytes):
    """Return the largest detected face, aligned and on the model's device, or None."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    boxes, _ = embedder.detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return None
    idx = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
    faces = embedder.detector.extract(img, boxes, save_path=None)
    return faces[idx : idx + 1].to(embedder.device)


def _rgb(t):
    """Convert one aligned face tensor back to an 8-bit RGB array for display."""
    arr = np.transpose(t.detach().squeeze().cpu().numpy(), (1, 2, 0))
    return (arr * 128.0 + 127.5).clip(0, 255).astype(np.uint8)


# inferno colormap anchors (black -> purple -> red -> orange -> pale yellow).
_INFERNO = (
    np.array(
        [
            [0, 0, 4],
            [40, 11, 84],
            [101, 21, 110],
            [159, 42, 99],
            [212, 72, 66],
            [245, 125, 21],
            [250, 193, 39],
            [252, 255, 164],
        ],
        dtype=np.float32,
    )
    / 255
)


def heat_overlay(rgb, sal, vmax=None, alpha=0.6):
    """Overlay an inferno heatmap on an RGB face, scaled to a shared vmax."""
    v = vmax or float(sal.max() + 1e-9)
    x = np.clip(sal / v, 0, 1) * (len(_INFERNO) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(_INFERNO) - 1)
    f = (x - lo)[..., None]
    cmap = _INFERNO[lo] * (1 - f) + _INFERNO[hi] * f  # HxWx3 in [0,1]
    out = (rgb / 255.0) * (1 - alpha) + cmap * alpha
    return (out * 255).clip(0, 255).astype(np.uint8)


def _saliency(attr):
    """Reduce an attribution tensor to one magnitude per pixel."""
    # per-pixel attribution magnitude, mean over channels (raw; the caller
    # scales both maps to a shared vmax).
    return attr.detach().abs().mean(dim=1).squeeze().cpu().numpy().astype(np.float32)


def explain(embedder, a_bytes, b_bytes, steps: int = 50):
    """Attribute the match score to image regions via integrated gradients."""
    a = _aligned(embedder, a_bytes)
    b = _aligned(embedder, b_bytes)
    if a is None or b is None:
        return None

    with torch.no_grad():
        e_a = embedder.model(a)
        e_b = embedder.model(b)
    sim = float(torch.cosine_similarity(e_a, e_b, dim=1).item())

    def toward(fixed):
        """Score the varying image by its similarity to the fixed embedding."""
        return lambda x: torch.cosine_similarity(embedder.model(x), fixed, dim=1)

    sal_a = _saliency(
        IntegratedGradients(toward(e_b)).attribute(
            a, baselines=torch.zeros_like(a), n_steps=steps
        )
    )
    sal_b = _saliency(
        IntegratedGradients(toward(e_a)).attribute(
            b, baselines=torch.zeros_like(b), n_steps=steps
        )
    )

    return {
        "similarity": sim,
        "a": _rgb(a),
        "sal_a": sal_a,
        "b": _rgb(b),
        "sal_b": sal_b,
    }
