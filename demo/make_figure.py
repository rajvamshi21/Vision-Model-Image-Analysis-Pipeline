#!/usr/bin/env python3
"""Render docs/review-grid.png: a scored contact sheet for the README."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa.pipeline import AnalysisPipeline  # noqa: E402
from vqa.quality.vlm import NullCritic  # noqa: E402

TILE, PAD, COLS = 230, 14, 4
COLOURS = {"pass": (26, 127, 55), "review": (154, 103, 0), "fail": (207, 34, 46)}
PICKS = ["_clean", "_blur", "_dark", "_noise", "_clutter", "_tiny", "_offcenter", "_blown"]


def font(size: int):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main() -> int:
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
    paths = []
    for suffix in PICKS:
        matches = sorted(data.glob(f"*{suffix}.jpg"))
        if matches:
            paths.append(matches[0])
    if not paths:
        print("No sample images found; run demo/generate_sample_images.py first")
        return 1

    analyses = AnalysisPipeline(critic=NullCritic()).analyse_paths(paths)
    rows = (len(analyses) + COLS - 1) // COLS
    width = COLS * TILE + (COLS + 1) * PAD
    height = rows * (TILE + 54) + (rows + 1) * PAD
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font, small_font = font(15), font(12)

    for index, analysis in enumerate(analyses):
        col, row = index % COLS, index // COLS
        x = PAD + col * (TILE + PAD)
        y = PAD + row * (TILE + 54 + PAD)
        tile = Image.open(analysis.record.source_uri).convert("RGB").resize((TILE, TILE),
                                                                           Image.LANCZOS)
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y, x + TILE, y + TILE], outline=(208, 215, 222))

        report = analysis.report
        colour = COLOURS[report.verdict]
        draw.rectangle([x, y + TILE + 6, x + 58, y + TILE + 26], fill=colour)
        draw.text((x + 8, y + TILE + 9), report.verdict.upper(), font=small_font,
                  fill=(255, 255, 255))
        draw.text((x + 66, y + TILE + 8), f"{report.score:.0f}/100", font=title_font,
                  fill=(31, 35, 40))
        codes = ", ".join(i.code for i in report.issues[:2]) or "no defects"
        while draw.textlength(codes, font=small_font) > TILE - 4 and len(codes) > 4:
            codes = codes[:-5] + "..."
        draw.text((x, y + TILE + 32), codes, font=small_font, fill=(87, 96, 106))

    out = Path("docs/review-grid.png")
    out.parent.mkdir(exist_ok=True)
    sheet.save(out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
