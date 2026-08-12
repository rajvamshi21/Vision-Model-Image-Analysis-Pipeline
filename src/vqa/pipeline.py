"""Orchestration: files (or bytes) in, analysed + embedded records out.

Stage order matters for cost. Cheap deterministic work first (decode,
heuristics), then one batched forward pass through the encoder, then the
optional -- and by far most expensive -- per-image VLM call, which is only made
when it can change the outcome.
"""

from __future__ import annotations

import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image

from vqa.config import Settings, get_settings
from vqa.embedding import Encoder, get_encoder
from vqa.imageio import downscale, load_image, to_array
from vqa.quality.heuristics import analyse_technical
from vqa.quality.scoring import DEFAULT_CONFIG, ScoringConfig, build_report
from vqa.quality.vlm import NullCritic, VLMCritic, get_critic
from vqa.quality.zeroshot import ZeroShotAttributeScorer, maybe_build_scorer
from vqa.types import ImageAnalysis, ImageRecord

logger = logging.getLogger(__name__)


@dataclass
class _Prepared:
    record: ImageRecord
    array: np.ndarray
    image: Image.Image


class AnalysisPipeline:
    """Stateless per-call; the encoder and critic are reused across batches."""

    def __init__(
        self,
        settings: Settings | None = None,
        encoder: Encoder | None = None,
        critic: VLMCritic | None = None,
        config: ScoringConfig = DEFAULT_CONFIG,
        vlm_review_below: float | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.encoder = encoder or get_encoder(self.settings)
        self.critic = critic if critic is not None else get_critic(self.settings)
        self.config = config
        # Only spend a VLM call when the cheap signals are ambiguous.
        self.vlm_review_below = (
            config.pass_threshold if vlm_review_below is None else vlm_review_below
        )
        self.zero_shot: ZeroShotAttributeScorer | None = maybe_build_scorer(self.encoder)
        if self.zero_shot is None:
            logger.info(
                "Encoder %s has no text tower: zero-shot attributes disabled "
                "(set VQA_EMBEDDING_BACKEND=openclip to enable)", self.encoder.name
            )

    # ---- preparation -----------------------------------------------------

    def _prepare_path(self, path: Path, sku: str | None = None) -> _Prepared:
        raw = path.read_bytes()
        return self._prepare_bytes(raw, source_uri=str(path), filename=path.name, sku=sku)

    def _prepare_bytes(self, raw: bytes, source_uri: str, filename: str,
                       sku: str | None = None) -> _Prepared:
        image = load_image(io.BytesIO(raw))
        width, height = image.size
        small = downscale(image, self.settings.max_side)
        record = ImageRecord(
            content_hash=hashlib.sha256(raw).hexdigest(),
            source_uri=source_uri,
            filename=filename,
            width=width,
            height=height,
            size_bytes=len(raw),
            format=(Image.open(io.BytesIO(raw)).format or "UNKNOWN"),
            sku=sku,
            metadata={"aspect_ratio": round(width / height, 4) if height else None},
        )
        return _Prepared(record=record, array=to_array(small), image=small)

    # ---- analysis --------------------------------------------------------

    def _analyse_prepared(self, items: Sequence[_Prepared]) -> list[ImageAnalysis]:
        if not items:
            return []

        metrics = [analyse_technical(it.array, it.record.width, it.record.height) for it in items]

        embeddings = self.encoder.encode_images([it.image for it in items])
        attributes = (
            self.zero_shot.score(embeddings) if self.zero_shot is not None
            else [None] * len(items)
        )

        # Provisional score decides who is worth a VLM call.
        provisional = [
            build_report(m, a, None, self.config) for m, a in zip(metrics, attributes, strict=True)
        ]
        critiques: list[object | None] = [None] * len(items)
        if not isinstance(self.critic, NullCritic):
            targets = [i for i, r in enumerate(provisional) if r.score < self.vlm_review_below]
            if targets:
                workers = max(1, min(self.settings.max_workers, len(targets)))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for idx, result in zip(
                        targets, pool.map(lambda i: self.critic.assess(items[i].image), targets),
                        strict=True,
                    ):
                        critiques[idx] = result

        analyses: list[ImageAnalysis] = []
        for i, item in enumerate(items):
            report = (
                provisional[i] if critiques[i] is None
                else build_report(metrics[i], attributes[i], critiques[i], self.config)
            )
            analyses.append(ImageAnalysis(
                record=item.record,
                report=report,
                embedding=embeddings[i],
                embedding_model=self.encoder.name,
            ))
        return analyses

    # ---- public API ------------------------------------------------------

    def analyse_paths(self, paths: Sequence[Path], skus: Sequence[str | None] | None = None,
                      ) -> list[ImageAnalysis]:
        """Analyse a batch of files; decoding is threaded, encoding is batched."""
        skus = list(skus) if skus is not None else [None] * len(paths)
        workers = max(1, min(self.settings.max_workers, len(paths) or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            prepared = list(pool.map(self._prepare_path, paths, skus))
        return self._analyse_prepared(prepared)

    def analyse_bytes(self, raw: bytes, filename: str = "upload",
                      source_uri: str = "upload") -> ImageAnalysis:
        return self._analyse_prepared([self._prepare_bytes(raw, source_uri, filename)])[0]

    def iter_batches(self, paths: Sequence[Path], skus: Sequence[str | None] | None = None,
                     ) -> Iterable[list[ImageAnalysis]]:
        """Yield results batch by batch so long runs stream to the sink."""
        skus = list(skus) if skus is not None else [None] * len(paths)
        size = max(1, self.settings.batch_size)
        for start in range(0, len(paths), size):
            yield self.analyse_paths(paths[start : start + size], skus[start : start + size])

    def embed_text(self, text: str) -> np.ndarray:
        if not self.encoder.supports_text:
            raise ValueError(
                f"Text search needs a CLIP-style encoder; {self.encoder.name} is image-only. "
                "Set VQA_EMBEDDING_BACKEND=openclip."
            )
        return self.encoder.encode_texts([text])[0]
