#!/usr/bin/env python3
"""How strong does a defect have to be before the pipeline notices?

A pass/fail benchmark on obviously-broken images only proves the detectors are
not asleep. This sweeps each degradation continuously over a clean image and
reports the point at which the corresponding issue first fires -- the operating
point you would actually quote to a stakeholder ("we catch blur from a
Gaussian radius of ~1.5px on a 1100px frame").

    python demo/sensitivity.py --data data/sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa.imageio import to_array  # noqa: E402
from vqa.quality.heuristics import analyse_technical  # noqa: E402
from vqa.quality.scoring import build_report  # noqa: E402


def _blur(image, level):
    return image if level == 0 else image.filter(ImageFilter.GaussianBlur(level))


def _gain(image, level):
    return Image.fromarray(
        np.clip(np.asarray(image, np.float32) * level, 0, 255).astype(np.uint8)
    )


def _noise(image, level):
    arr = np.asarray(image, np.float32)
    rng = np.random.default_rng(0)
    return Image.fromarray(
        np.clip(arr + rng.normal(0, level, arr.shape), 0, 255).astype(np.uint8)
    )


def _shift(image, level):
    canvas = Image.new("RGB", image.size, (250, 250, 252))
    scaled = image.resize((int(image.width * 0.62), int(image.height * 0.62)), Image.LANCZOS)
    dx = int(image.width * level)
    canvas.paste(scaled, ((image.width - scaled.width) // 2 + dx,
                          (image.height - scaled.height) // 2))
    return canvas


def _downscale(image, level):
    return image.resize((int(level), int(level)), Image.LANCZOS)


SWEEPS = [
    ("blurry", "Gaussian blur radius (px)", _blur,
     [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], False),
    ("underexposed", "Exposure gain", _gain,
     [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2], False),
    ("overexposed", "Exposure gain", _gain,
     [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2], False),
    ("noisy", "Additive noise sigma (0-255)", _noise,
     [0, 4, 8, 12, 16, 20, 26, 32], False),
    ("off_center", "Horizontal offset (frame fraction)", _shift,
     [0.0, 0.05, 0.08, 0.10, 0.14, 0.18, 0.25], False),
    ("low_resolution", "Short side (px)", _downscale,
     [1100, 1000, 900, 850, 800, 700, 500], True),
]


def analyse(image: Image.Image):
    metrics = analyse_technical(to_array(image), image.width, image.height)
    return build_report(metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/sample")
    parser.add_argument("--samples", type=int, default=6, help="clean base images to average over")
    args = parser.parse_args()

    clean = sorted(Path(args.data).glob("*_clean.jpg"))[: args.samples]
    if not clean:
        print(f"No *_clean.jpg in {args.data}; run demo/generate_sample_images.py first",
              file=sys.stderr)
        return 1
    bases = [Image.open(p).convert("RGB") for p in clean]

    print(f"Averaged over {len(bases)} clean base images\n")
    print("| Defect | Parameter | First detected at | Score at that point |")
    print("|---|---|---|---|")

    for code, label, fn, levels, descending in SWEEPS:
        firsts, scores = [], []
        for base in bases:
            for level in levels:
                report = analyse(fn(base, level))
                if any(i.code == code for i in report.issues):
                    firsts.append(level)
                    scores.append(report.score)
                    break
        if not firsts:
            print(f"| {code} | {label} | not reached within sweep | - |")
            continue
        aggregate = min(firsts) if descending else max(firsts)
        detail = "" if len(set(firsts)) == 1 else f" (range {min(firsts)}-{max(firsts)})"
        print(f"| {code} | {label} | {aggregate}{detail} "
              f"| {np.mean(scores):.0f}/100 |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
