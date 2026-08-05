"""Face embedder: MTCNN detection + InceptionResnetV1 (VGGFace2) 512-dim embedding from image bytes."""

import os
import warnings
from io import BytesIO

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# TORCH_HOME points weight downloads at a writable dir; torch reads it lazily at
# model load, so it can follow the imports.
os.environ.setdefault("TORCH_HOME", os.environ.get("TORCH_HOME", "/tmp/torch"))
warnings.filterwarnings("ignore")

EMBEDDING_DIM = 512


class FaceEmbedder:
    """MTCNN face detection + InceptionResnetV1 (VGGFace2) 512-dim embedding."""

    def __init__(self, device: str | None = None):
        # CPU by default (matches the Lambda runtime); MPS/CUDA optional locally.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # keep_all=True so we can choose the largest face ourselves.
        self.detector = MTCNN(keep_all=True, device=self.device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def get_embedding(
        self, image_bytes: bytes, strategy: str = "largest"
    ) -> np.ndarray | None:
        """Return a 512-dim embedding from raw image bytes, or None if no face.

        strategy:
            "largest": pick the largest detected face by bbox area (default).
            "first": pick the highest-confidence detection.
        """
        try:
            if not image_bytes:
                return None
            img = Image.open(BytesIO(image_bytes)).convert("RGB")

            boxes, _ = self.detector.detect(img)
            if boxes is None or len(boxes) == 0:
                return None

            if strategy == "largest":
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                idx = int(np.argmax(areas))
            else:
                idx = 0

            # Aligned, standardized 160x160 face tensors for the detected boxes.
            faces = self.detector.extract(img, boxes, save_path=None)
            if faces is None:
                return None

            face = faces[idx : idx + 1].to(self.device)  # keep the batch dim
            with torch.no_grad():
                embedding = self.model(face)

            return embedding.squeeze(0).cpu().numpy()

        except (OSError, ValueError, RuntimeError, TypeError, IndexError):
            # a bad/undecodable image or a detection with no usable face -> no
            # embedding; real bugs (bad weights, config) still surface.
            return None
