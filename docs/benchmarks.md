# Benchmarks

Every number here is produced by scripts in this repo and is reproducible in
about a minute on a laptop CPU:

```bash
python demo/generate_sample_images.py --out data/sample --count 120
python demo/benchmark.py    --data data/sample     # accuracy + retrieval
python demo/sensitivity.py  --data data/sample     # detection thresholds
```

## The corpus, and what it is not

Real e-commerce imagery cannot be redistributed, so the repo ships a
**generator** rather than a dataset. It renders 14 clean studio-style packshots
(6 product shapes × colour variants) and applies 8 deterministic degradations to
each, giving 126 images with ground-truth defect labels in `manifest.csv`.

This is a synthetic, self-labelled corpus. It is good for what it is used for —
regression-testing the detectors, calibrating thresholds, and making the
pipeline's operating point explicit — and it is **not** evidence of accuracy on
real photographs. Two honest caveats:

- The thresholds in `vqa.quality.heuristics` were fitted by inspecting the
  metric distributions of this corpus. Perfect recall on the same corpus is
  therefore partly circular. The held-out column below re-runs the frozen
  thresholds against a corpus generated from a different random seed.
- Verdicts are defect-driven (`any high-severity issue → fail`), so
  "defect catch rate" follows closely from per-defect recall. The two numbers
  are not independent evidence.

## Defect detection

| Defect | Calibration set (seed 7) | Held out (seed 99) |
|---|---|---|
| `blurry` | 1.00 | 1.00 |
| `underexposed` | 1.00 | 1.00 |
| `overexposed` | 1.00 | 0.93 |
| `noisy` | 1.00 | 1.00 |
| `cluttered_background` | 1.00 | 1.00 |
| `subject_too_small` | 1.00 | 1.00 |
| `off_center` | 1.00 | 1.00 |
| `low_resolution` | 1.00 | 1.00 |
| **Clean images passed** | 14/14 (1.00) | 14/14 (1.00) |
| **False alarms on clean images** | 0 | 0 |
| **Balanced accuracy** | 1.00 | 0.996 |
| Mean score, clean | 88.9 | 88.9 |
| Mean score, degraded | 64.4 | 64.4 |

The single held-out miss is a mild over-exposure that lands just under the
`subject_luminance > 0.68` trigger — visible in the sensitivity sweep below,
where over-exposure is the least sensitive detector.

## Sensitivity: how bad does it have to get?

Each degradation is swept continuously over 6 clean 1100 px base images; the
table reports where the corresponding issue first fires.

| Defect | Parameter | First detected at | Score at that point |
|---|---|---|---|
| `blurry` | Gaussian blur radius | 1.0–1.5 px | 64/100 |
| `underexposed` | Exposure gain | 0.5× | 71/100 |
| `overexposed` | Exposure gain | 1.5–2.0× | 89/100 |
| `noisy` | Additive noise σ (0–255) | 16 | 72/100 |
| `off_center` | Horizontal offset | 0.14 of frame width | 81/100 |
| `low_resolution` | Short side | 700 px (rule: < 800 px) | 85/100 |

A 1.5 px blur on a 1100 px frame is roughly "slightly soft at 100% zoom" — the
sort of thing a human reviewer catches on a second pass. Over-exposure is the
weakest detector and is the obvious next thing to improve.

## Retrieval

Nearest-neighbour quality of the **dependency-free perceptual encoder**
(`perceptual-hash-v1`, the CI default). Each of the 112 degraded images queries
the other 125; the corpus has 8 same-SKU images per query, so chance precision
is 0.064.

| Metric | Calibration | Held out | Chance |
|---|---|---|---|
| Same-SKU precision@5 | 0.502 | 0.491 | 0.064 |
| Same-SKU top-1 | 0.580 | 0.589 | 0.064 |
| Clean sibling in top-5 | 0.661 | 0.661 | 0.040 |

That is ~8× chance from 512 dimensions of layout, colour and gradient
statistics with no learned weights — a deliberately weak but honest baseline.
It fails exactly where you would expect: the `tiny` and `offcenter` variants
change global layout, which is all this descriptor sees. Switching to
`VQA_EMBEDDING_BACKEND=openclip` replaces it with CLIP/SigLIP features, which
are scale- and position-tolerant; no CLIP numbers are published here because
they have not been run on this corpus.

## Throughput

126 images (1100×1100 JPEG) end to end — decode, downscale, 14 measurements,
512-d embedding — on 2 CPU cores, no GPU:

```
6.8 images/s  ·  148 ms/image  ·  batch size 16
```

The VLM stage is excluded: it is off by default and, when enabled, only runs on
images that provisionally score below the pass threshold — on this corpus, 81%
of the degraded images and 0% of the clean ones, i.e. it pays for reasoning
exactly where the cheap signals are already unhappy.

## Reproducing the tables

```bash
python demo/generate_sample_images.py --out /tmp/holdout --count 120 --seed 99
python demo/benchmark.py --data /tmp/holdout --json holdout.json
```

CI runs the generator and `demo/benchmark.py` on every push and uploads
`metrics.json` as an artifact, so drift in these numbers shows up as a diff.
