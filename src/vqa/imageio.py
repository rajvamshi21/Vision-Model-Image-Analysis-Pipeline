"""Image loading helpers shared by the pipeline and the demo scripts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: str | Path) -> Image.Image:
    """Open an image, honour EXIF orientation and drop alpha onto white."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image, mask=image.split()[-1])
        return canvas
    return image.convert("RGB")


def downscale(image: Image.Image, max_side: int) -> Image.Image:
    """Bound the longest side so analysis cost is independent of input size."""
    longest = max(image.size)
    if max_side <= 0 or longest <= max_side:
        return image
    ratio = max_side / float(longest)
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(size, Image.LANCZOS)


def to_array(image: Image.Image) -> np.ndarray:
    """RGB float32 array in [0, 1] with shape (H, W, 3)."""
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def iter_image_paths(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
