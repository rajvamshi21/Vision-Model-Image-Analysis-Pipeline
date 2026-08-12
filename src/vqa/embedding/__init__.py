"""Encoder factory with graceful degradation.

If the ``models`` extra is not installed (or the weights cannot be fetched) the
pipeline falls back to the dependency-free perceptual encoder and logs why,
rather than failing the whole run.
"""

from __future__ import annotations

import logging

from vqa.config import Settings, get_settings
from vqa.embedding.base import Encoder
from vqa.embedding.hashing import PerceptualHashEncoder

logger = logging.getLogger(__name__)

__all__ = ["Encoder", "PerceptualHashEncoder", "get_encoder"]

_CACHE: dict[tuple, Encoder] = {}


def get_encoder(settings: Settings | None = None, *, cache: bool = True) -> Encoder:
    settings = settings or get_settings()
    key = (settings.embedding_backend, settings.embedding_model,
           settings.embedding_pretrained, settings.device)
    if cache and key in _CACHE:
        return _CACHE[key]

    backend = settings.embedding_backend
    encoder: Encoder
    if backend in ("openclip", "clip", "siglip"):
        try:
            from vqa.embedding.clip_encoder import OpenClipEncoder

            encoder = OpenClipEncoder(
                settings.embedding_model,
                settings.embedding_pretrained,
                device=settings.device,
                batch_size=settings.batch_size,
            )
        except Exception as exc:  # pragma: no cover - depends on optional deps
            logger.warning(
                "Falling back to the perceptual encoder (%s unavailable: %s). "
                "Install with: pip install 'vqa[models]'",
                settings.embedding_model, exc,
            )
            encoder = PerceptualHashEncoder()
    elif backend == "hash":
        encoder = PerceptualHashEncoder()
    else:
        raise ValueError(f"Unknown VQA_EMBEDDING_BACKEND: {backend!r}")

    if cache:
        _CACHE[key] = encoder
    return encoder
