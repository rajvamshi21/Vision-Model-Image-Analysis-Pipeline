"""Encoder interface shared by every embedding backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class Encoder(ABC):
    """Maps images (and optionally text) into one shared L2-normalised space."""

    name: str = "encoder"
    dim: int = 512
    supports_text: bool = False

    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return an (n, dim) float32 array of unit-norm row vectors."""

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not support text queries")

    @staticmethod
    def l2_normalise(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        return matrix / np.maximum(norms, 1e-8)
