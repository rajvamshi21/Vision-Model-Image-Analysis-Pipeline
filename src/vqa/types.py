"""Plain dataclasses shared by the analysis, storage and API layers.

Deliberately dependency-free (no pydantic) so the core library installs with
just numpy + pillow; the API layer converts these to JSON at the edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TechnicalMetrics:
    """Signal-level measurements plus their normalised 0..1 sub-scores."""

    width: int
    height: int
    sharpness_raw: float          # mean of the strongest 20% |Laplacian| on the subject
    acutance: float               # sharpness_raw normalised by subject contrast
    noise_sigma: float
    mean_luminance: float
    subject_luminance: float
    subject_contrast: float
    clipped_highlights: float
    clipped_shadows: float
    contrast_raw: float
    colorfulness_raw: float
    background_std: float
    background_luminance: float
    subject_coverage: float
    center_offset: float

    sharpness: float
    exposure: float
    contrast: float
    noise: float
    background: float
    composition: float
    resolution: float

    def subscores(self) -> dict[str, float]:
        return {
            "sharpness": self.sharpness,
            "exposure": self.exposure,
            "contrast": self.contrast,
            "noise": self.noise,
            "background": self.background,
            "composition": self.composition,
            "resolution": self.resolution,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticAttributes:
    """Zero-shot CLIP probabilities for human-readable attributes."""

    model: str
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VLMAssessment:
    """Structured critique returned by a multimodal model."""

    provider: str
    model: str
    score: float | None = None
    caption: str | None = None
    tags: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    raw: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str          # "low" | "medium" | "high"
    message: str
    remedy: str
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityReport:
    score: float
    verdict: str           # "pass" | "review" | "fail"
    technical: TechnicalMetrics
    issues: list[Issue] = field(default_factory=list)
    semantic: SemanticAttributes | None = None
    vlm: VLMAssessment | None = None
    pipeline_version: str = "scoring-v3"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "technical": self.technical.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "semantic": self.semantic.to_dict() if self.semantic else None,
            "vlm": self.vlm.to_dict() if self.vlm else None,
            "pipeline_version": self.pipeline_version,
        }


@dataclass(frozen=True)
class ImageRecord:
    content_hash: str
    source_uri: str
    filename: str
    width: int
    height: int
    size_bytes: int
    format: str
    sku: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageAnalysis:
    record: ImageRecord
    report: QualityReport
    embedding: Any = None          # numpy.ndarray, L2-normalised
    embedding_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": asdict(self.record),
            "report": self.report.to_dict(),
            "embedding_model": self.embedding_model,
            "embedding_dim": None if self.embedding is None else int(self.embedding.shape[-1]),
        }


@dataclass(frozen=True)
class SearchHit:
    image_id: int
    source_uri: str
    filename: str
    sku: str | None
    similarity: float
    score: float | None
    verdict: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
