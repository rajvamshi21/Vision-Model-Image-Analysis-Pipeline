"""Signal-level image quality measurements.

Every measurement is a small, explainable numpy kernel rather than a learned
model: results are deterministic, cost nothing to run, and give the report a
reason it can show a human ("edge acutance 0.07 vs 1.21 for a sharp shot").
The learned models (CLIP zero-shot, VLM critique) sit on top of this layer.

Two design decisions matter more than the individual kernels:

1. **Measure the subject, not the frame.** A packshot on a white sweep is 88%
   background; global mean luminance says "overexposed" for every good image.
   So we segment a foreground mask first and compute exposure, contrast and
   sharpness inside it.
2. **Normalise sharpness by contrast.** Raw edge energy conflates "soft focus"
   with "dark" and "low contrast". Dividing by subject contrast isolates focus.

Calibration constants were fitted on the labelled synthetic set in
``demo/generate_sample_images.py`` -- see docs/benchmarks.md.
"""

from __future__ import annotations

import numpy as np

from vqa.types import TechnicalMetrics

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# --- calibration constants (see docs/benchmarks.md) ------------------------
EDGE_PERCENTILE = 0.20        # share of subject pixels treated as "edges"
ACUTANCE_LOG_MIN = -1.20      # log10(acutance) mapped to sub-score 0.0
ACUTANCE_LOG_MAX = 0.10       # ... and to 1.0
NOISE_SIGMA_MAX = 0.06
SUBJECT_CONTRAST_TARGET = 0.10
EXPOSURE_PLATEAU = (0.28, 0.66)
EXPOSURE_FALLOFF = 0.20
BACKGROUND_STD_MAX = 0.20
RESOLUTION_MIN = 200.0
RESOLUTION_GOOD = 1200.0
IDEAL_COVERAGE = (0.08, 0.75)
COVERAGE_FALLOFF = 0.25
CENTER_TOLERANCE = 0.22
MIN_MASK_PIXELS = 200


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA


def abs_laplacian(gray: np.ndarray) -> np.ndarray:
    """|4-neighbour Laplacian| -- high where the image has crisp detail."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros((1, 1), dtype=np.float32)
    return np.abs(
        4.0 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    )


def edge_energy(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean of the strongest ``EDGE_PERCENTILE`` Laplacian responses.

    Averaging over the whole frame would let a large flat background drown out
    a small in-focus product, so we rank and keep the top slice instead.
    """
    lap = abs_laplacian(gray)
    region = lap.ravel()
    if mask is not None:
        inner = mask[1:-1, 1:-1]
        if inner.sum() >= MIN_MASK_PIXELS:
            region = lap[inner]
    if region.size == 0:
        return 0.0
    k = max(16, int(EDGE_PERCENTILE * region.size))
    k = min(k, region.size)
    return float(np.mean(np.partition(region, -k)[-k:]))


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast estimator: a kernel blind to smooth image structure,
    so what survives is mostly sensor noise."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    conv = (
        4.0 * gray[1:-1, 1:-1]
        - 2.0 * (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
        + gray[:-2, :-2] + gray[:-2, 2:] + gray[2:, :-2] + gray[2:, 2:]
    )
    scale = np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2))
    return float(np.abs(conv).sum() * scale)


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness, in [0, 1] image units."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    """Otsu's between-class variance threshold over a 1-D distribution."""
    top = float(values.max()) if values.size else 0.0
    hist, edges = np.histogram(values, bins=bins, range=(0.0, top or 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_b = np.cumsum(hist)
    weight_f = total - weight_b
    valid = (weight_b > 0) & (weight_f > 0)
    if not valid.any():
        return float(centers[-1])
    cumsum = np.cumsum(hist * centers)
    mean_b = np.divide(cumsum, weight_b, out=np.zeros_like(cumsum), where=weight_b > 0)
    mean_f = np.divide(cumsum[-1] - cumsum, weight_f, out=np.zeros_like(cumsum), where=weight_f > 0)
    between = weight_b * weight_f * (mean_b - mean_f) ** 2
    between[~valid] = -1.0
    return float(centers[int(np.argmax(between))])


def border_band(rgb: np.ndarray) -> np.ndarray:
    """Pixels in a thin frame around the edge -- the background proxy."""
    h, w = rgb.shape[:2]
    band = max(2, int(round(0.03 * min(h, w))))
    parts = [rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3)]
    left, right = rgb[band:-band, :band], rgb[band:-band, -band:]
    if left.size:
        parts.append(left.reshape(-1, 3))
    if right.size:
        parts.append(right.reshape(-1, 3))
    return np.concatenate(parts, axis=0)


def subject_mask(rgb: np.ndarray, background_color: np.ndarray) -> np.ndarray:
    """Foreground = pixels far enough from the estimated background colour.

    Otsu picks the split point, clamped so a perfectly clean sweep does not
    produce a degenerate all-or-nothing mask.
    """
    distance = np.linalg.norm(rgb - background_color[None, None, :], axis=-1)
    threshold = float(np.clip(otsu_threshold(distance), 0.06, 0.35))
    return distance > threshold


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Centre of the mask's bounding box, normalised, robust to stray specks."""
    if not mask.any():
        return None
    row_idx = np.flatnonzero(mask.mean(axis=1) > 0.01)
    col_idx = np.flatnonzero(mask.mean(axis=0) > 0.01)
    if row_idx.size == 0 or col_idx.size == 0:
        return None
    cy = (row_idx[0] + row_idx[-1]) / 2.0 / max(1, mask.shape[0] - 1)
    cx = (col_idx[0] + col_idx[-1]) / 2.0 / max(1, mask.shape[1] - 1)
    return float(cx), float(cy)


def _plateau_score(value: float, low: float, high: float, falloff: float) -> float:
    """1.0 inside [low, high], decaying linearly outside over ``falloff``."""
    if value < low:
        return _clip01(1.0 - (low - value) / falloff)
    if value > high:
        return _clip01(1.0 - (value - high) / falloff)
    return 1.0


def analyse_technical(rgb: np.ndarray, width: int, height: int) -> TechnicalMetrics:
    """Run every measurement and normalise the results to 0..1 sub-scores.

    ``rgb`` is the (possibly downscaled) analysis array; ``width``/``height``
    are the *original* dimensions so resolution is judged on the source file.
    """
    gray = luminance(rgb)

    border = border_band(rgb)
    bg_color = np.median(border, axis=0).astype(np.float32)
    bg_std = float(np.linalg.norm(border.std(axis=0)))
    bg_lum = float(bg_color @ LUMA)

    mask = subject_mask(rgb, bg_color)
    coverage = float(mask.mean())
    subject = gray[mask] if mask.sum() >= MIN_MASK_PIXELS else gray.ravel()
    subject_lum = float(subject.mean())
    subject_contrast = float(subject.std())

    sharp_raw = edge_energy(gray, mask)
    acutance = sharp_raw / max(subject_contrast, 1e-6)
    sigma = noise_sigma(gray)

    centroid = mask_centroid(mask)
    center_offset = 0.0 if centroid is None else float(
        np.hypot(centroid[0] - 0.5, centroid[1] - 0.5)
    )

    # --- normalised sub-scores -------------------------------------------
    log_acutance = np.log10(max(acutance, 1e-9))
    sharpness = _clip01(
        (log_acutance - ACUTANCE_LOG_MIN) / (ACUTANCE_LOG_MAX - ACUTANCE_LOG_MIN)
    )
    exposure = _plateau_score(subject_lum, *EXPOSURE_PLATEAU, EXPOSURE_FALLOFF)
    contrast = _clip01(subject_contrast / SUBJECT_CONTRAST_TARGET)
    noise = _clip01(1.0 - sigma / NOISE_SIGMA_MAX)
    background = _clip01(1.0 - bg_std / BACKGROUND_STD_MAX) * _clip01(0.45 + 0.55 * bg_lum)

    coverage_score = _plateau_score(coverage, *IDEAL_COVERAGE, COVERAGE_FALLOFF)
    if coverage < IDEAL_COVERAGE[0]:
        # Below the band, fall off proportionally rather than linearly: a
        # product filling 0.3% of the frame is far worse than one at 6%.
        coverage_score = _clip01(coverage / IDEAL_COVERAGE[0])
    centering_score = _clip01(1.0 - center_offset / CENTER_TOLERANCE)
    composition = 0.6 * coverage_score + 0.4 * centering_score

    min_side = float(min(width, height))
    resolution = _clip01(
        np.log2(max(min_side, 1.0) / RESOLUTION_MIN) / np.log2(RESOLUTION_GOOD / RESOLUTION_MIN)
    )

    return TechnicalMetrics(
        width=width,
        height=height,
        sharpness_raw=sharp_raw,
        acutance=acutance,
        noise_sigma=sigma,
        mean_luminance=float(gray.mean()),
        subject_luminance=subject_lum,
        subject_contrast=subject_contrast,
        clipped_highlights=float((gray > 0.98).mean()),
        clipped_shadows=float((gray < 0.02).mean()),
        contrast_raw=float(gray.std()),
        colorfulness_raw=colorfulness(rgb),
        background_std=bg_std,
        background_luminance=bg_lum,
        subject_coverage=coverage,
        center_offset=center_offset,
        sharpness=sharpness,
        exposure=exposure,
        contrast=contrast,
        noise=noise,
        background=background,
        composition=composition,
        resolution=resolution,
    )
