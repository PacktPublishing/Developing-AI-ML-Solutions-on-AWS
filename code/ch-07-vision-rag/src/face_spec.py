"""The chapter's inference contract, served two ways.

One `InferenceSpec` defines what it means to embed, register, and match a face. Locally
ModelBuilder serves it with Mode.IN_PROCESS, which is the only local path that reaches
an Apple GPU; on AWS the same class runs inside the serving container behind an
asynchronous SageMaker endpoint on a GPU instance. The code that answers is identical,
which is the parity claim this chapter can actually make.

    {"op": "register",   "keys": [...]}                          # batched, the GPU case
    {"op": "match",   "key": "probes/x.jpg", "claim": "..."}  # 1:N verification
    {"op": "compare", "a": "a.jpg", "b": "b.jpg", "explain": true}
"""

import json
import os
import uuid
from io import BytesIO
from pathlib import Path

from sagemaker.serve import InferenceSpec

# torch, psycopg and the model live behind the methods that need them. Importing this
# module should cost nothing but the interface: aws/deploy.py constructs the spec just
# to hand its shape to ModelBuilder, and has no reason to install PyTorch to do it.

MATCH = float(os.environ.get("MATCH_THRESHOLD", "0.70"))


def _ref(params: dict, name: str) -> str:
    """Return one required object reference, naming it plainly when it is missing.

    Without this a malformed request reaches boto3 or the filesystem as None and
    fails there, so the caller gets a ParamValidationError or a TypeError instead
    of being told which field it left out.
    """
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{name}' is required and must be an object key")
    return value


def read_image(ref: str) -> bytes:
    """Read an image by S3 key on AWS, or by path locally.

    IMAGES_S3_BUCKET is set only where there is a bucket, which is what makes the same
    spec work against a local folder and an object store without branching elsewhere.
    """
    if bucket := os.environ.get("IMAGES_S3_BUCKET"):
        import boto3

        return boto3.client("s3").get_object(Bucket=bucket, Key=ref)["Body"].read()
    root = Path(os.environ.get("FACES_DIR", "data/generated/faces"))
    return (root / ref).read_bytes()


class FaceEmbeddingSpec(InferenceSpec):
    """Load FaceNet on the best available device and answer one request."""

    def load(self, model_dir: str | None = None):
        """Load the model once, when the server starts."""
        from face_embedder import FaceEmbedder, best_devices

        device, detector_device = best_devices()
        embedder = FaceEmbedder(device=device, detector_device=detector_device)
        print(f"model on {embedder.device}, detector on {embedder.detector_device}")
        return embedder

    def invoke(self, input_object: dict | str | bytes, model) -> dict:
        """Dispatch one request to register, match, or compare.

        The two servers deliver the payload differently: the container parses the JSON
        body and passes a dict, while the SDK's in-process server passes whatever its
        deserializer produced, which is the raw string. Accept either.
        """
        if isinstance(input_object, str | bytes | bytearray):
            input_object = json.loads(input_object)
        op = input_object.get("op", "match")
        handler = {
            "register": self._register,
            "match": self._match,
            "compare": self._compare,
        }
        if op not in handler:
            return {"error": f"unknown op {op!r}"}
        result = handler[op](input_object, model)
        result["device"] = str(model.device)
        return result

    def _register(self, params: dict, model) -> dict:
        """Embed many photos in one batch and store them: the operation a GPU rewards."""
        import kycstore

        keys = params.get("keys", [])
        # A caller that passes one space-separated string instead of a list gets an
        # opaque NoSuchKey from S3 on a key that is eight paths glued together. Say
        # what is wrong instead, and accept the string.
        if isinstance(keys, str):
            keys = keys.split()
        if not all(isinstance(k, str) and k for k in keys):
            return {"error": "keys must be a list of object keys", "batch": len(keys)}
        vectors = model.get_embeddings([read_image(k) for k in keys])
        registered, skipped = 0, []
        with kycstore.connect() as conn:
            kycstore.ensure_schema(conn)
            for key, vector in zip(keys, vectors):
                if vector is None:
                    skipped.append(key)
                    continue
                parts = key.split("/")  # registered/{subject}/{file}
                kycstore.insert(
                    conn, parts[1] if len(parts) > 2 else parts[0], key, vector
                )
                registered += 1
        return {"registered": registered, "skipped": skipped, "batch": len(keys)}

    def _match(self, params: dict, model) -> dict:
        """1:N search, plus the claim that turns a search into a verification."""
        import kycstore

        key, claim = _ref(params, "key"), params.get("claim")
        vector = model.get_embedding(read_image(key))
        if vector is None:
            return {"key": key, "matches": [], "identified": None, "matched": False}
        with kycstore.connect() as conn:
            matches = kycstore.search(
                conn, vector, int(params.get("k", 5)), exclude_key=key
            )
        top = matches[0] if matches else None
        identified = top["subject"] if top else None
        return {
            "key": key,
            "matches": matches,
            "identified": identified,
            "matched": bool(
                top and top["score"] >= MATCH and (claim is None or identified == claim)
            ),
        }

    def _compare(self, params: dict, model) -> dict:
        """1:1 similarity, with an optional Captum explanation."""
        import numpy as np

        a, b = _ref(params, "a"), _ref(params, "b")
        a_bytes, b_bytes = read_image(a), read_image(b)
        ea, eb = model.get_embedding(a_bytes), model.get_embedding(b_bytes)
        if ea is None or eb is None:
            return {"a": a, "b": b, "similarity": None}

        result = {"a": a, "b": b, "similarity": round(float(np.dot(ea, eb)), 4)}
        if params.get("explain"):
            png = self._explain_png(model, a_bytes, b_bytes)
            if png is not None:
                result["explanation_key"] = self._write_png(png)
        return result

    @staticmethod
    def _explain_png(model, a_bytes: bytes, b_bytes: bytes) -> bytes | None:
        """Four panels: each face beside its Integrated Gradients heatmap."""
        from face_explainer import explain, heat_overlay
        from PIL import Image

        out = explain(model, a_bytes, b_bytes)
        if out is None:
            return None
        vmax = max(float(out["sal_a"].max()), float(out["sal_b"].max()))
        panels = [
            out["a"],
            heat_overlay(out["a"], out["sal_a"], vmax),
            out["b"],
            heat_overlay(out["b"], out["sal_b"], vmax),
        ]
        cell = 256
        canvas = Image.new("RGB", (cell * 4, cell), (30, 30, 30))
        for i, rgb in enumerate(panels):
            canvas.paste(
                Image.fromarray(rgb).resize((cell, cell), Image.Resampling.LANCZOS),
                (i * cell, 0),
            )
        buf = BytesIO()
        canvas.save(buf, "PNG")
        return buf.getvalue()

    @staticmethod
    def _write_png(png: bytes) -> str:
        """Put the explanation where the caller can fetch it."""
        key = f"explanations/{uuid.uuid4().hex}.png"
        if bucket := os.environ.get("IMAGES_S3_BUCKET"):
            import boto3

            boto3.client("s3").put_object(
                Bucket=bucket, Key=key, Body=png, ContentType="image/png"
            )
            return key
        path = Path("/tmp") / Path(key).name
        path.write_bytes(png)
        return str(path)
