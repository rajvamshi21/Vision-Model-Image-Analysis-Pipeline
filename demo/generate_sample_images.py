#!/usr/bin/env python3
"""Generate a labelled synthetic product-photo corpus.

Real e-commerce imagery cannot be redistributed, so the repo ships a generator
instead: clean studio-style packshots plus deterministic degradations whose
ground-truth defect labels are written to ``manifest.csv``. That turns "the
scorer looks reasonable" into a measurable precision/recall number
(``demo/benchmark.py``) and keeps CI hermetic.

    python demo/generate_sample_images.py --out data/sample --count 120
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CANVAS = 1100
PALETTE = [
    ((214, 74, 62), "crimson"), ((44, 110, 190), "cobalt"), ((36, 148, 118), "jade"),
    ((228, 168, 54), "amber"), ((122, 88, 190), "violet"), ((58, 62, 74), "graphite"),
]
CATEGORIES = ("bottle", "box", "mug", "can", "headphones", "tube")

# degradation -> (expected issue codes, should the image pass?)
DEGRADATIONS: dict[str, tuple[tuple[str, ...], bool]] = {
    "clean":       ((), True),
    "blur":        (("blurry",), False),
    "dark":        (("underexposed",), False),
    "noise":       (("noisy",), False),
    "clutter":     (("cluttered_background",), False),
    "tiny":        (("subject_too_small",), False),
    "offcenter":   (("off_center",), False),
    "lowres":      (("low_resolution",), False),
    "blown":       (("overexposed",), False),
}


def _shade(color, factor):
    return tuple(int(max(0, min(255, c * factor))) for c in color)


def _vertical_gradient(size, top, bottom):
    height = size[1]
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top_arr = np.array(top, dtype=np.float32)[None, None, :]
    bottom_arr = np.array(bottom, dtype=np.float32)[None, None, :]
    column = top_arr * (1 - ramp[:, :, None]) + bottom_arr * ramp[:, :, None]
    return Image.fromarray(np.repeat(column, size[0], axis=1).astype(np.uint8), "RGB")


def _paste_shaded(canvas, mask, color):
    """Paste a body-coloured, vertically shaded shape through ``mask``."""
    gradient = _vertical_gradient(canvas.size, _shade(color, 1.25), _shade(color, 0.62))
    canvas.paste(gradient, (0, 0), mask)


def _draw_product(canvas: Image.Image, category: str, color, rng: random.Random) -> None:
    scale = 2
    size = (canvas.width * scale, canvas.height * scale)
    mask = Image.new("L", size, 0)
    detail = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    dd = ImageDraw.Draw(detail)
    w, h = size
    cx = w // 2

    if category == "bottle":
        d.rounded_rectangle([cx - w // 8, int(h * 0.34), cx + w // 8, int(h * 0.84)],
                            radius=w // 40, fill=255)
        d.rectangle([cx - w // 22, int(h * 0.20), cx + w // 22, int(h * 0.36)], fill=255)
        d.rounded_rectangle([cx - w // 18, int(h * 0.15), cx + w // 18, int(h * 0.22)],
                            radius=w // 90, fill=255)
        dd.rectangle([cx - w // 9, int(h * 0.50), cx + w // 9, int(h * 0.68)], fill=255)
    elif category == "box":
        d.polygon([(cx - w // 6, int(h * 0.36)), (cx + w // 8, int(h * 0.30)),
                   (cx + w // 8, int(h * 0.74)), (cx - w // 6, int(h * 0.80))], fill=255)
        d.polygon([(cx + w // 8, int(h * 0.30)), (cx + w // 4, int(h * 0.36)),
                   (cx + w // 4, int(h * 0.72)), (cx + w // 8, int(h * 0.74))], fill=255)
        dd.polygon([(cx - w // 6, int(h * 0.36)), (cx + w // 8, int(h * 0.30)),
                    (cx + w // 8, int(h * 0.42)), (cx - w // 6, int(h * 0.48))], fill=255)
    elif category == "mug":
        d.rounded_rectangle([cx - w // 8, int(h * 0.36), cx + w // 8, int(h * 0.76)],
                            radius=w // 60, fill=255)
        d.ellipse([cx + w // 10, int(h * 0.44), cx + w // 4, int(h * 0.64)], fill=255)
        d.ellipse([cx + w // 8, int(h * 0.48), cx + w // 5, int(h * 0.60)], fill=0)
        dd.ellipse([cx - w // 8, int(h * 0.34), cx + w // 8, int(h * 0.40)], fill=255)
    elif category == "can":
        d.rounded_rectangle([cx - w // 10, int(h * 0.30), cx + w // 10, int(h * 0.78)],
                            radius=w // 70, fill=255)
        d.ellipse([cx - w // 10, int(h * 0.27), cx + w // 10, int(h * 0.34)], fill=255)
        dd.rectangle([cx - w // 10, int(h * 0.46), cx + w // 10, int(h * 0.58)], fill=255)
    elif category == "headphones":
        d.arc([cx - w // 5, int(h * 0.26), cx + w // 5, int(h * 0.66)], 180, 360, fill=255,
              width=w // 34)
        d.rounded_rectangle([cx - w // 5 - w // 26, int(h * 0.44), cx - w // 5 + w // 26,
                             int(h * 0.66)], radius=w // 50, fill=255)
        d.rounded_rectangle([cx + w // 5 - w // 26, int(h * 0.44), cx + w // 5 + w // 26,
                             int(h * 0.66)], radius=w // 50, fill=255)
        dd.arc([cx - w // 5, int(h * 0.26), cx + w // 5, int(h * 0.66)], 200, 340, fill=255,
               width=w // 90)
    else:  # tube
        d.rounded_rectangle([cx - w // 9, int(h * 0.30), cx + w // 9, int(h * 0.80)],
                            radius=w // 30, fill=255)
        d.polygon([(cx - w // 9, int(h * 0.32)), (cx + w // 9, int(h * 0.32)),
                   (cx + w // 30, int(h * 0.22)), (cx - w // 30, int(h * 0.22))], fill=255)
        dd.rectangle([cx - w // 9, int(h * 0.52), cx + w // 9, int(h * 0.62)], fill=255)

    mask = mask.resize(canvas.size, Image.LANCZOS)
    detail = detail.resize(canvas.size, Image.LANCZOS)

    # Contact shadow, then the shaded body, then a lighter label band.
    shadow = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(shadow).ellipse(
        [canvas.width * 0.30, canvas.height * 0.79, canvas.width * 0.70, canvas.height * 0.86],
        fill=90,
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(canvas.width / 80))
    canvas.paste((205, 205, 208), (0, 0), shadow)

    _paste_shaded(canvas, mask, color)
    canvas.paste(_shade(color, 1.55), (0, 0), Image.eval(detail, lambda v: int(v * 0.75)))

    # Fine surface texture keeps the frequency content photo-like.
    rng_texture = np.random.default_rng(rng.randrange(1 << 30))
    texture = rng_texture.normal(0, 3.0, (canvas.height, canvas.width, 1))
    weight = np.asarray(mask, np.float32)[..., None] / 255.0
    arr = np.asarray(canvas, dtype=np.float32) + texture * weight
    canvas.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB"))


def make_base(category: str, color, rng: random.Random) -> Image.Image:
    canvas = _vertical_gradient((CANVAS, CANVAS), (255, 255, 255), (243, 243, 246))
    _draw_product(canvas, category, color, rng)
    return canvas


def apply_degradation(image: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32)

    if kind == "clean":
        return image
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(rng.uniform(5.0, 9.0)))
    if kind == "dark":
        return Image.fromarray(np.clip(arr * rng.uniform(0.22, 0.32), 0, 255).astype(np.uint8))
    if kind == "blown":
        return Image.fromarray(np.clip(arr * rng.uniform(1.6, 2.0) + 40, 0, 255).astype(np.uint8))
    if kind == "noise":
        generator = np.random.default_rng(rng.randrange(1 << 30))
        noise = generator.normal(0, rng.uniform(26, 38), arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    if kind == "clutter":
        cluttered = image.copy()
        draw = ImageDraw.Draw(cluttered, "RGBA")
        for _ in range(90):
            x, y = rng.randrange(CANVAS), rng.randrange(CANVAS)
            r = rng.randrange(20, 110)
            tone = rng.randrange(40, 190)
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=(tone, rng.randrange(40, 190), rng.randrange(40, 190), 130))
        # Keep the product readable: clutter is a background problem.
        return Image.blend(cluttered, image, 0.35)
    if kind == "tiny":
        small = image.resize((int(CANVAS * 0.16), int(CANVAS * 0.16)), Image.LANCZOS)
        canvas = Image.new("RGB", (CANVAS, CANVAS), (252, 252, 253))
        canvas.paste(small, ((CANVAS - small.width) // 2, (CANVAS - small.height) // 2))
        return canvas
    if kind == "offcenter":
        canvas = Image.new("RGB", (CANVAS, CANVAS), (250, 250, 252))
        dx = int(CANVAS * rng.uniform(0.22, 0.30)) * rng.choice([-1, 1])
        dy = int(CANVAS * rng.uniform(0.10, 0.18)) * rng.choice([-1, 1])
        canvas.paste(image.resize((int(CANVAS * 0.62), int(CANVAS * 0.62)), Image.LANCZOS),
                     (CANVAS // 2 - int(CANVAS * 0.31) + dx, CANVAS // 2 - int(CANVAS * 0.31) + dy))
        return canvas
    if kind == "lowres":
        return image.resize((360, 360), Image.LANCZOS)
    raise ValueError(f"unknown degradation {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/sample")
    parser.add_argument("--count", type=int, default=120, help="approximate number of images")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    kinds = list(DEGRADATIONS)
    n_bases = max(1, math.ceil(args.count / len(kinds)))
    rows = []

    for index in range(n_bases):
        category = CATEGORIES[index % len(CATEGORIES)]
        color, color_name = PALETTE[(index // len(CATEGORIES)) % len(PALETTE)]
        sku = f"{category}-{color_name}-{index:02d}"
        base = make_base(category, color, rng)

        for kind in kinds:
            image = apply_degradation(base, kind, rng)
            name = f"{sku}_{kind}.jpg"
            image.save(out / name, quality=94, subsampling=0)
            expected, should_pass = DEGRADATIONS[kind]
            rows.append({
                "filename": name,
                "sku": sku,
                "category": category,
                "color": color_name,
                "degradation": kind,
                "expected_issues": ";".join(expected),
                "should_pass": int(should_pass),
            })

    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} images and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
