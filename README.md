# Vision QA Pipeline

**Automated quality assessment for product photography, with pgvector similarity retrieval over the image corpus.**

Point it at a folder of product images. It scores every image, explains what is
wrong in language a photographer can act on, and indexes a 512-d embedding in
Postgres so you can ask *"show me other listings that look like this one."*

[![CI](https://github.com/rajvamshi21/Vision-Model-Image-Analysis-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/rajvamshi21/Vision-Model-Image-Analysis-Pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![Scored review grid](docs/review-grid.png)

---

## What it does

- **Scores** each image 0–100 across seven dimensions (focus, exposure,
  contrast, noise, background, framing, resolution) and returns a
  `pass` / `review` / `fail` verdict.
- **Explains** every verdict as a list of defects with severities and fixes —
  `blurry (high): edge acutance 0.07; a sharp packshot is ~1.2 → reshoot on a
  tripod`. No opaque single number.
- **Indexes** a CLIP/SigLIP embedding per image in `pgvector` with an HNSW
  cosine index, for near-duplicate detection, "find this product's other
  photos", and text→image search.
- **Escalates** the ambiguous cases to a multimodal model (Claude / GPT) for a
  written critique — and only those, so the API bill tracks the hard tail.

## Why it is built this way

The interesting decisions are not "call a model", they are the ones that make
the output trustworthy and the bill small.

**Measure the subject, not the frame.** A packshot on a white sweep is ~85%
background. Global mean luminance says *overexposed* for every good image — the
first version of this scorer failed 100% of clean photos for exactly that
reason. The pipeline segments a foreground mask (Otsu on distance-from-border
colour) and computes exposure, contrast and sharpness inside it.

**Normalise sharpness by contrast.** Raw Laplacian variance conflates soft
focus with "dark" and "low contrast": an underexposed photo measures as blurry.
Dividing edge energy by subject contrast isolates focus — the underexposed
sample below scores 0.88 on sharpness and 0.04 on exposure, which is the truth.

**Rank the edges, don't average them.** Mean Laplacian over the whole frame lets
a large flat background drown out a small, perfectly sharp product. The metric
is the mean of the strongest 20% of responses *inside the subject mask*.

**One bad dimension should sink an image.** A pure weighted mean is too
forgiving — a completely out-of-focus photo still averages 70/100 because
everything else is fine. The score mixes in the worst sub-score
(`0.7 · mean + 0.3 · min`), and the verdict is defect-driven first: any
high-severity issue fails the image regardless of the average.

**Spend inference where it changes the answer.** Heuristics are free and run on
everything. The encoder runs batched, once. The VLM runs only on images that
provisionally score below the pass threshold — 81% of defective images and 0%
of clean ones on the benchmark corpus.

**Degrade, never crash.** No GPU or `torch`? The encoder falls back to a
dependency-free 512-d perceptual descriptor and logs why. No API key? The VLM
stage is a no-op. No Postgres? `vqa analyze` still works. The core library needs
only `numpy` and `pillow`.

## Quickstart

No GPU, no database, no API key — about 60 seconds:

```bash
git clone https://github.com/rajvamshi21/Vision-Model-Image-Analysis-Pipeline
cd Vision-Model-Image-Analysis-Pipeline
pip install -e .

python demo/generate_sample_images.py --out data/sample --count 120
vqa analyze data/sample
```

```
bottle-cobalt-06_blur.jpg   50.4/100  FAIL
    sharpness    #....................... 0.04
    exposure     ######################## 1.00
    contrast     ######################## 1.00
    noise        ######################## 1.00
    background   ####################.... 0.84
    composition  ######################## 0.99
    resolution   #######################. 0.95
    [high  ] blurry: Image is soft (edge acutance 0.07; a sharp packshot is ~1.2).
             -> Reshoot on a tripod or raise shutter speed; avoid upscaling small originals.
```

Then the interactive review queue:

```bash
pip install -e ".[demo]" && streamlit run demo/app.py
```

## With Postgres + pgvector

```bash
pip install -e ".[db,api]"
docker compose up -d db          # pgvector/pgvector:pg16 on :5433

vqa init-db
vqa ingest data/sample --sku-from-name
vqa search --image data/sample/bottle-cobalt-06_clean.jpg -k 5
vqa stats
```

```
   sim   score  verdict  file
 1.000    74.8  fail     bottle-cobalt-06_lowres.jpg
 0.999    50.4  fail     bottle-cobalt-06_blur.jpg
 0.998    66.5  review   bottle-cobalt-06_noise.jpg
```

Enable real CLIP embeddings and multimodal critique:

```bash
pip install -e ".[models,vlm]"
export VQA_EMBEDDING_BACKEND=openclip     # ViT-B-32, 512-d, downloads once
export VQA_VLM_PROVIDER=anthropic         # + ANTHROPIC_API_KEY
vqa ingest data/sample --force
vqa search --text "a red bottle on a white background"
```

## Architecture

![Architecture](docs/architecture.svg)

Storage is three tables so each concern can change independently: `images`
(metadata, SHA-256 dedupe key), `image_embeddings` (`vector(512)` + HNSW cosine
index, rebuildable when you swap encoders), and `quality_reports` (JSONB,
versioned by `pipeline_version` so a scoring change is auditable rather than
destructive). Similarity search runs in Postgres via the `<=>` operator, so
retrieval stays correct well past what fits in memory.

## How the score is composed

| Sub-score | Weight | Measurement |
|---|---|---|
| `sharpness` | 0.28 | Top-20% \|Laplacian\| inside the subject mask, normalised by subject contrast |
| `exposure` | 0.16 | Subject luminance against a plateau (0.28–0.66), with linear falloff |
| `contrast` | 0.12 | Subject luminance standard deviation |
| `noise` | 0.12 | Immerkær σ estimate (structure-blind high-pass kernel) |
| `background` | 0.14 | Border-band colour variance × background brightness |
| `composition` | 0.12 | Subject coverage plateau (0.08–0.75) + bounding-box centring |
| `resolution` | 0.06 | Log-scaled short side of the **original** file |

```
technical = 0.7 · weighted_mean + 0.3 · worst_sub_score
score     = blend(technical, CLIP zero-shot, VLM critique)   # renormalised when a signal is absent
verdict   = fail if any high-severity issue or score < 55
            review if any issue or score < 75
            pass otherwise
```

Weights and every threshold live in `ScoringConfig` — a marketplace with
different framing rules changes a dataclass, not the detectors.

### Output

```jsonc
{
  "image": { "content_hash": "c17cadad…", "width": 1100, "height": 1100, "format": "JPEG" },
  "report": {
    "score": 50.4,
    "verdict": "fail",
    "technical": { "acutance": 0.0704, "sharpness": 0.037, "exposure": 1.0, "background": 0.842 },
    "issues": [{
      "code": "blurry",
      "severity": "high",
      "message": "Image is soft (edge acutance 0.07; a sharp packshot is ~1.2).",
      "remedy": "Reshoot on a tripod or raise shutter speed; avoid upscaling small originals.",
      "value": 0.0366
    }],
    "pipeline_version": "scoring-v3"
  },
  "embedding_model": "perceptual-hash-v1",
  "embedding_dim": 512
}
```

## Results

Measured on a labelled synthetic corpus generated by
`demo/generate_sample_images.py` (126 images, 8 defect classes), with thresholds
frozen and re-run against a corpus from a different seed. Full tables, method
and caveats: **[docs/benchmarks.md](docs/benchmarks.md)**.

| | Calibration | Held out (unseen seed) |
|---|---|---|
| Per-defect recall (8 classes) | 1.00 | 0.99 (one mild over-exposure missed) |
| Clean images passed | 14/14 | 14/14 |
| False alarms on clean images | 0 | 0 |
| Mean score, clean vs degraded | 88.9 / 64.4 | 88.9 / 64.4 |
| Same-SKU retrieval precision@5 | 0.502 | 0.491 (chance: 0.064) |
| Throughput (2 CPU cores, no GPU) | 6.8 img/s · 148 ms/image | — |

The corpus is synthetic and the thresholds were fitted on it, so these numbers
describe the pipeline's *operating point*, not its accuracy on real
photographs. `demo/sensitivity.py` makes that operating point explicit: blur is
caught from a 1.0–1.5 px Gaussian radius on a 1100 px frame, noise from σ ≈ 16,
off-centring from 14% of the frame width.

## REST API

```bash
uvicorn vqa.api:app --port 8000    # or: vqa serve
```

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Active encoder, dimensionality, VLM provider |
| `POST` | `/analyze` | Score an upload, store nothing |
| `POST` | `/index` | Score an upload and write it to pgvector |
| `GET` | `/images`, `/images/{id}` | Review queue, worst first |
| `GET` | `/search/similar/{id}` | Neighbours of an indexed image |
| `POST` | `/search/by-image` | Neighbours of an upload |
| `GET` | `/search/by-text?q=` | Text→image search (CLIP backend only; `501` otherwise) |
| `GET` | `/stats` | Verdict distribution and the most common defects |

## CLI

| Command | Purpose |
|---|---|
| `vqa analyze PATH…` | Score images, print or `--json` the reports. No database. |
| `vqa ingest PATH` | Analyse and index a folder; `--dry-run`, `--force`, `--sku-from-name` |
| `vqa search` | `--image` / `--text` / `--id`, filtered by `--verdict` and `--min-score` |
| `vqa init-db` · `vqa stats` · `vqa config` | Schema, corpus summary, resolved settings |

## Configuration

All settings are environment variables (see `.env.example`), so the same image
runs locally, in Docker and in CI.

| Variable | Default | Notes |
|---|---|---|
| `VQA_DATABASE_URL` | `postgresql://vqa:vqa@localhost:5433/vqa` | |
| `VQA_EMBEDDING_BACKEND` | `hash` | `openclip` for real CLIP/SigLIP features |
| `VQA_EMBEDDING_MODEL` / `_PRETRAINED` | `ViT-B-32` / `laion2b_s34b_b79k` | 512-d, matches the schema |
| `VQA_VLM_PROVIDER` | `none` | `anthropic` \| `openai` |
| `VQA_BATCH_SIZE` / `_MAX_WORKERS` / `_MAX_SIDE` | `16` / `8` / `1024` | Throughput knobs |

> Changing the embedding dimensionality also means changing `vector(512)` in
> `db/init/001_schema.sql` and rebuilding the HNSW index.

## Layout

```
src/vqa/
  imageio.py            decode · EXIF · alpha flattening · downscale
  pipeline.py           orchestration: prepare → heuristics → encode → critique → score
  quality/
    heuristics.py       subject segmentation and the 14 raw measurements
    scoring.py          weights, thresholds, issue rules, verdicts
    zeroshot.py         contrastive prompt pairs over the CLIP text tower
    vlm.py              Anthropic / OpenAI critics with structured JSON output
  embedding/
    clip_encoder.py     OpenCLIP / SigLIP
    hashing.py          dependency-free 512-d fallback descriptor
  repository.py         pgvector upserts and cosine search
  api.py · cli.py       FastAPI service and CLI
demo/                   corpus generator · benchmark · sensitivity sweep · Streamlit UI
tests/                  44 tests, no network, no database, no model downloads
```

## Limitations and next steps

- **Over-exposure is the weakest detector** — it fires at ~1.5× gain and missed
  one held-out case. Highlight clipping measured inside the subject mask would
  be a better signal than mean subject luminance.
- **The fallback encoder is layout-based**, so it fails on scale and position
  changes (`tiny`, `offcenter` in the benchmark). That is the honest baseline;
  CLIP features fix it, and the backend swap is one environment variable.
- **No real-photo evaluation.** The obvious next step is a few hundred labelled
  real listing photos to validate the thresholds outside the generator.
- **Ingestion is single-process.** For millions of images the pipeline wants a
  work queue and `COPY`-based bulk loading rather than per-image upserts.
- **`quality_reports` is versioned but never garbage-collected** — old
  `pipeline_version` rows accumulate by design; a retention policy is missing.

## Licence

MIT — see [LICENSE](LICENSE).
