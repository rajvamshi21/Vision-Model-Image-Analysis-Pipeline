"""OpenCLIP / SigLIP image+text encoder (optional extra: ``pip install "vqa[models]"``)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from vqa.embedding.base import Encoder


def _resolve_device(requested: str) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenClipEncoder(Encoder):
    """Wraps an open_clip model; weights are downloaded once and cached."""

    supports_text = True

    def __init__(self, model_name: str, pretrained: str, device: str = "auto",
                 batch_size: int = 16) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.name = f"open_clip/{model_name}/{pretrained}"
        with torch.no_grad():
            probe = self.model.encode_text(self.tokenizer(["probe"]).to(self.device))
        self.dim = int(probe.shape[-1])

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            batch = torch.stack([self.preprocess(img.convert("RGB")) for img in chunk])
            with torch.no_grad():
                features = self.model.encode_image(batch.to(self.device))
            out.append(features.float().cpu().numpy())
        return self.l2_normalise(np.concatenate(out, axis=0))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return self.l2_normalise(features.float().cpu().numpy())
