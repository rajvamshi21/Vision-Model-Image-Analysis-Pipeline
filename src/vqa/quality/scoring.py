"""Turn measurements into a score, a verdict and an actionable issue list.

The scorer is deliberately transparent: a weighted sum over named sub-scores
plus threshold rules that each produce a human-readable remedy. Weights and
thresholds live in :class:`ScoringConfig` so they can be tuned per marketplace
without touching the detectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vqa import PIPELINE_VERSION
from vqa.types import Issue, QualityReport, SemanticAttributes, TechnicalMetrics, VLMAssessment

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class ScoringConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "sharpness": 0.28,
            "exposure": 0.16,
            "contrast": 0.12,
            "noise": 0.12,
            "background": 0.14,
            "composition": 0.12,
            "resolution": 0.06,
        }
    )
    worst_weight: float = 0.30         # how much the worst sub-score drags the mean
    semantic_weight: float = 0.30      # share given to CLIP zero-shot when available
    vlm_weight: float = 0.25           # share given to the VLM critique when available
    pass_threshold: float = 75.0
    review_threshold: float = 55.0

    # Detector thresholds (all measured on the subject, not the whole frame)
    blur_threshold: float = 0.42
    noise_threshold: float = 0.45
    contrast_threshold: float = 0.45
    background_threshold: float = 0.55
    dark_subject: float = 0.26
    dark_background: float = 0.55       # a truly underexposed frame is dark everywhere
    bright_subject: float = 0.68
    bright_background: float = 0.95
    noisy_background_std: float = 0.15  # above this, clutter is real and not grain
    min_coverage: float = 0.03
    max_coverage: float = 0.86
    max_center_offset: float = 0.14
    min_short_side: int = 800
    max_aspect_deviation: float = 0.6   # |log2(w/h)|


DEFAULT_CONFIG = ScoringConfig()

# Zero-shot attributes that count towards the semantic half of the score.
POSITIVE_SEMANTIC_KEYS = (
    "in_focus",
    "professional_lighting",
    "clean_background",
    "well_framed",
    "studio_quality",
)


def technical_score(metrics: TechnicalMetrics, config: ScoringConfig = DEFAULT_CONFIG) -> float:
    """Weighted mean, pulled down by the worst single sub-score.

    A pure weighted mean is too forgiving: an unusable, completely out-of-focus
    photo still scores ~0.73 because every other dimension is fine. Mixing in
    the minimum sub-score means one catastrophic failure sinks the image, which
    is how a human reviewer actually behaves.
    """
    subs = metrics.subscores()
    total = sum(config.weights.values())
    if total <= 0:
        return 0.0
    mean = sum(subs[k] * w for k, w in config.weights.items()) / total
    worst = min(subs.values())
    return (1.0 - config.worst_weight) * mean + config.worst_weight * worst


def semantic_score(attributes: SemanticAttributes | None) -> float | None:
    if attributes is None or not attributes.scores:
        return None
    values = [attributes.scores[k] for k in POSITIVE_SEMANTIC_KEYS if k in attributes.scores]
    if not values:
        return None
    return float(sum(values) / len(values))


def composite_score(
    metrics: TechnicalMetrics,
    attributes: SemanticAttributes | None = None,
    vlm: VLMAssessment | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> float:
    """Blend available signals, renormalising when a signal is missing."""
    parts: list[tuple[float, float]] = [(technical_score(metrics, config), 1.0)]

    sem = semantic_score(attributes)
    if sem is not None:
        parts.append((sem, config.semantic_weight))

    if vlm is not None and vlm.score is not None and vlm.error is None:
        parts.append((max(0.0, min(1.0, vlm.score / 100.0)), config.vlm_weight))

    # The technical term keeps whatever weight is left over.
    extra = sum(w for _, w in parts[1:])
    parts[0] = (parts[0][0], max(0.0, 1.0 - extra))
    total_weight = sum(w for _, w in parts) or 1.0
    blended = sum(value * weight for value, weight in parts) / total_weight
    return round(100.0 * max(0.0, min(1.0, blended)), 1)


def verdict_for(
    score: float,
    issues: list[Issue] | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> str:
    """Verdict is defect-driven first, score-driven second.

    A listing gate cares about *what* is wrong, not only about an average: one
    high-severity defect fails the image even if everything else is perfect.
    The score then ranks the queue a human reviews.
    """
    severities = {i.severity for i in (issues or [])}
    if "high" in severities or score < config.review_threshold:
        return "fail"
    if severities or score < config.pass_threshold:
        return "review"
    return "pass"


def detect_issues(
    metrics: TechnicalMetrics,
    config: ScoringConfig = DEFAULT_CONFIG,
    vlm: VLMAssessment | None = None,
) -> list[Issue]:
    """Threshold rules -> explainable, deduplicated defect list."""
    issues: list[Issue] = []

    def add(code: str, severity: str, message: str, remedy: str, value: float) -> None:
        issues.append(Issue(code=code, severity=severity, message=message,
                            remedy=remedy, value=round(float(value), 4)))

    if metrics.sharpness < config.blur_threshold:
        severity = "high" if metrics.sharpness < config.blur_threshold * 0.6 else "medium"
        add("blurry", severity,
            f"Image is soft (edge acutance {metrics.acutance:.2f}; a sharp packshot is ~1.2).",
            "Reshoot on a tripod or raise shutter speed; avoid upscaling small originals.",
            metrics.sharpness)

    if metrics.noise < config.noise_threshold:
        add("noisy", "medium",
            f"Visible sensor noise (sigma {metrics.noise_sigma:.3f}).",
            "Lower ISO, add light, or denoise before publishing.",
            metrics.noise)

    dark = (metrics.subject_luminance < config.dark_subject
            and metrics.background_luminance < config.dark_background)
    bright = (metrics.subject_luminance > config.bright_subject
              and metrics.background_luminance > config.bright_background)
    if dark:
        add("underexposed", "high",
            f"Frame is dark (subject luminance {metrics.subject_luminance:.2f}).",
            "Add fill light or raise exposure by roughly one stop.",
            metrics.subject_luminance)
    elif bright:
        add("overexposed", "medium",
            f"Product is washed out (subject luminance {metrics.subject_luminance:.2f}, "
            f"{metrics.clipped_highlights:.0%} of the frame clipped).",
            "Reduce exposure or diffuse the key light to recover product detail.",
            metrics.subject_luminance)

    if metrics.contrast < config.contrast_threshold:
        add("low_contrast", "low",
            f"Flat tonal range on the product (subject std {metrics.subject_contrast:.3f}).",
            "Increase lighting separation between product and background.",
            metrics.contrast)

    # Grain in the background is already reported as noise -- do not double-count it.
    grain_only = ("noisy" in {i.code for i in issues}
                  and metrics.background_std < config.noisy_background_std)
    if metrics.background < config.background_threshold and not grain_only:
        busy = metrics.background_std > 0.10
        add("cluttered_background", "high" if busy else "medium",
            f"Background is {'busy' if busy else 'dark'} (edge std {metrics.background_std:.3f}, "
            f"luminance {metrics.background_luminance:.2f}).",
            "Shoot on a clean white sweep or cut the product out.",
            metrics.background)

    if metrics.subject_coverage < config.min_coverage:
        add("subject_too_small", "medium",
            f"Product fills only {metrics.subject_coverage:.1%} of the frame.",
            "Crop tighter so the product occupies 25-70% of the image.",
            metrics.subject_coverage)
    elif metrics.subject_coverage > config.max_coverage:
        add("subject_cropped", "medium",
            f"Product fills {metrics.subject_coverage:.0%} of the frame and may be cut off.",
            "Leave a margin of at least 5% around the product.",
            metrics.subject_coverage)

    if metrics.center_offset > config.max_center_offset:
        add("off_center", "low",
            f"Product is off-centre by {metrics.center_offset:.2f} of the frame diagonal.",
            "Recentre the product; most marketplaces expect a centred subject.",
            metrics.center_offset)

    short_side = min(metrics.width, metrics.height)
    if short_side < config.min_short_side:
        add("low_resolution", "high" if short_side < config.min_short_side / 2 else "medium",
            f"Short side is {short_side}px, below the {config.min_short_side}px zoom requirement.",
            "Upload the original capture instead of a resized copy.",
            short_side)

    if metrics.height > 0:
        import math
        deviation = abs(math.log2(metrics.width / metrics.height))
        if deviation > config.max_aspect_deviation:
            add("extreme_aspect", "low",
                f"Aspect ratio {metrics.width}:{metrics.height} is far from square.",
                "Pad or crop to a 1:1 canvas for consistent grid rendering.",
                deviation)

    if vlm is not None and vlm.error is None:
        known = {i.code for i in issues}
        for text in vlm.issues:
            code = "vlm_" + "_".join(text.lower().split()[:3]).strip(".,")
            if code not in known:
                add(code, "low", text, "Reviewed by the multimodal critic.", 0.0)

    issues.sort(key=lambda i: SEVERITY_ORDER.get(i.severity, 3))
    return issues


def build_report(
    metrics: TechnicalMetrics,
    attributes: SemanticAttributes | None = None,
    vlm: VLMAssessment | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> QualityReport:
    score = composite_score(metrics, attributes, vlm, config)
    issues = detect_issues(metrics, config, vlm)
    return QualityReport(
        score=score,
        verdict=verdict_for(score, issues, config),
        technical=metrics,
        issues=issues,
        semantic=attributes,
        vlm=vlm,
        pipeline_version=PIPELINE_VERSION,
    )
