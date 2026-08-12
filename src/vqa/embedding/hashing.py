"""Dependency-free perceptual encoder.

Not a replacement for CLIP -- it has no semantics -- but it produces a real,
deterministic 512-d descriptor from layout, colour and gradient structure, so
the whole pipeline (ingest -> pgvector -> nearest-neighbour search) is runnable
and testable in CI without downloading multi-gigabyte weights. It is also a
useful baseline: "how much of my retrieval quality actually comes from CLIP?"

Layout: 256 dims spatial luminance (16x16) + 192 dims colour layout (8x8 RGB)
+ 32 dims gradient-orientation histogram + 32 dims intensity histogram.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vqa.embedding.base import Encoder

GRAY_GRID = 16
COLOR_GRID = 8
ORIENTATION_BINS = 32
INTENSITY_BINS = 32
BLOCK_WEIGHTS = (0.45, 0.25, 0.22, 0.08)


def _standardise(array: np.ndarray, axis=None) -> np.ndarray:
    """Zero-mean, unit-variance -- removes exposure and gain from the descriptor."""
    mean = array.mean(axis=axis, keepdims=axis is not None)
    std = array.std(axis=axis, keepdims=axis is not None)
    return (array - mean) / np.maximum(std, 1e-6)


class PerceptualHashEncoder(Encoder):
    name = "perceptual-hash-v1"
    dim = GRAY_GRID**2 + 3 * COLOR_GRID**2 + ORIENTATION_BINS + INTENSITY_BINS  # 512
    supports_text = False

    def _descriptor(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")

        gray = np.asarray(
            rgb.convert("L").resize((GRAY_GRID, GRAY_GRID), Image.LANCZOS), dtype=np.float32
        ) / 255.0
        # Z-scoring (not just mean-centring) makes the descriptor invariant to
        # linear brightness/contrast changes, so an under-exposed copy of a
        # product still retrieves its clean sibling.
        gray_block = _standardise(gray).ravel()

        color = np.asarray(
            rgb.resize((COLOR_GRID, COLOR_GRID), Image.LANCZOS), dtype=np.float32
        ) / 255.0
        color_block = _standardise(color, axis=(0, 1)).ravel()

        edges = rgb.convert("L").resize((64, 64), Image.LANCZOS)
        work = np.asarray(edges, dtype=np.float32) / 255.0
        gy, gx = np.gradient(work)
        magnitude = np.hypot(gx, gy).ravel()
        angle = (np.arctan2(gy, gx).ravel() + np.pi) % np.pi          # orientation, not direction
        orientation_block, _ = np.histogram(
            angle, bins=ORIENTATION_BINS, range=(0.0, np.pi), weights=magnitude
        )

        # Histogram of the standardised image: the shape of the tonal
        # distribution rather than absolute brightness.
        intensity_block, _ = np.histogram(
            _standardise(work), bins=INTENSITY_BINS, range=(-3.0, 3.0)
        )

        blocks = [
            gray_block,
            color_block,
            orientation_block.astype(np.float32),
            intensity_block.astype(np.float32),
        ]
        parts = []
        for block, weight in zip(blocks, BLOCK_WEIGHTS, strict=True):
            norm = float(np.linalg.norm(block))
            parts.append(block * (weight / norm) if norm > 1e-8 else block)
        return np.concatenate(parts).astype(np.float32)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.l2_normalise(np.stack([self._descriptor(img) for img in images]))
