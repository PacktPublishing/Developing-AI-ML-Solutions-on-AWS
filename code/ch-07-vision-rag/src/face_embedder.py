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


def best_devices() -> tuple[str, str]:
    """Return (embedding device, detector device) for this machine.

    MTCNN's image pyramid resamples at arbitrary scales, which the MPS backend has no
    adaptive-pool kernel for, so on Apple Silicon the detector stays on the CPU while
    the embedding model uses the GPU. Lambda has no accelerator, so both are the CPU
    there and the parity argument holds. Set FACE_DEVICE to force one device.
    """
    forced = os.environ.get("FACE_DEVICE")
    if forced:
        return forced, forced
    if torch.cuda.is_available():
        return "cuda", "cuda"
    if torch.backends.mps.is_available():
        return "mps", "cpu"
    return "cpu", "cpu"


class FaceEmbedder:
    """MTCNN face detection + InceptionResnetV1 (VGGFace2) 512-dim embedding."""

    def __init__(self, device: str | None = None, detector_device: str | None = None):
        """Build the embedder; detector_device runs MTCNN and defaults to device.

        Split them to use an Apple GPU: MTCNN's image pyramid resamples at arbitrary
        scales, which the MPS backend has no adaptive-pool kernel for, so it must stay
        on the CPU while the embedding model runs on "mps".
        """
        # CPU by default (matches the Lambda runtime); MPS/CUDA optional locally.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.detector_device = torch.device(detector_device or device)

        # keep_all=True so we can choose the largest face ourselves.
        self.detector = MTCNN(keep_all=True, device=self.detector_device)
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

    def get_embeddings(self, images: list[bytes]) -> list["np.ndarray | None"]:
        """Embed many images in one forward pass, keeping the order of the input.

        Detection is per image either way, but the embedding model is fed one stacked
        batch rather than one face at a time. That is the difference a GPU rewards:
        a single 160x160 face leaves the device almost idle, while a batch fills it.
        """
        aligned, slots = [], []
        for position, image_bytes in enumerate(images):
            try:
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                boxes, _ = self.detector.detect(img)
                if boxes is None or len(boxes) == 0:
                    continue
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                faces = self.detector.extract(img, boxes, save_path=None)
                if faces is None:
                    continue
                aligned.append(faces[int(np.argmax(areas))])
                slots.append(position)
            except (OSError, ValueError, RuntimeError, TypeError, IndexError):
                continue

        out: list[np.ndarray | None] = [None] * len(images)
        if not aligned:
            return out
        with torch.no_grad():
            batch = self.model(torch.stack(aligned).to(self.device)).cpu().numpy()
        for position, vector in zip(slots, batch):
            out[position] = vector
        return out
