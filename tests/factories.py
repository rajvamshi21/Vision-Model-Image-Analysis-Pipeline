"""Deterministic image builders used across the test suite."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from vqa.imageio import to_array
from vqa.quality.heuristics import analyse_technical


def product_image(size: int = 900, color=(200, 70, 60), offset: int = 0,
                  scale: float = 0.30) -> Image.Image:
    """A crisp coloured product on a near-white sweep."""
    canvas = Image.new("RGB", (size, size), (252, 252, 253))
    draw = ImageDraw.Draw(canvas)
    half = int(size * scale / 2)
    cx, cy = size // 2 + offset, size // 2
    draw.rounded_rectangle([cx - half, cy - int(half * 1.4), cx + half, cy + int(half * 1.4)],
                           radius=size // 40, fill=color)
    draw.rectangle([cx - half, cy - half // 3, cx + half, cy + half // 3],
                   fill=tuple(min(255, int(c * 1.5)) for c in color))
    return canvas


def blurred(image: Image.Image, radius: float = 6.0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius))


def darkened(image: Image.Image, gain: float = 0.25) -> Image.Image:
    return Image.fromarray(
        np.clip(np.asarray(image, np.float32) * gain, 0, 255).astype(np.uint8)
    )


def noisy(image: Image.Image, sigma: float = 30.0, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(image, np.float32) + rng.normal(0, sigma, (image.height, image.width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def metrics_for(image: Image.Image):
    return analyse_technical(to_array(image), image.width, image.height)


def as_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
