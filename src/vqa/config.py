"""Environment-driven configuration.

Everything is read from ``VQA_*`` environment variables so the same code runs
locally, in Docker and in CI without a config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_DATABASE_URL = "postgresql://vqa:vqa@localhost:5433/vqa"


def _str(key: str, default: str) -> str:
    value = os.getenv(key, "").strip()
    return value or default


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL

    # Embeddings
    embedding_backend: str = "hash"          # "openclip" | "hash"
    embedding_model: str = "ViT-B-32"
    embedding_pretrained: str = "laion2b_s34b_b79k"
    embedding_dim: int = 512
    device: str = "auto"

    # Multimodal critique
    vlm_provider: str = "none"               # "none" | "anthropic" | "openai"
    vlm_model: str = "claude-sonnet-5"
    vlm_max_tokens: int = 700

    # Runtime
    batch_size: int = 16
    max_workers: int = 8
    max_side: int = 1024                     # images are downscaled before analysis

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=_str("VQA_DATABASE_URL", DEFAULT_DATABASE_URL),
            embedding_backend=_str("VQA_EMBEDDING_BACKEND", "hash").lower(),
            embedding_model=_str("VQA_EMBEDDING_MODEL", "ViT-B-32"),
            embedding_pretrained=_str("VQA_EMBEDDING_PRETRAINED", "laion2b_s34b_b79k"),
            embedding_dim=_int("VQA_EMBEDDING_DIM", 512),
            device=_str("VQA_DEVICE", "auto"),
            vlm_provider=_str("VQA_VLM_PROVIDER", "none").lower(),
            vlm_model=_str("VQA_VLM_MODEL", "claude-sonnet-5"),
            vlm_max_tokens=_int("VQA_VLM_MAX_TOKENS", 700),
            batch_size=_int("VQA_BATCH_SIZE", 16),
            max_workers=_int("VQA_MAX_WORKERS", 8),
            max_side=_int("VQA_MAX_SIDE", 1024),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
