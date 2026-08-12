"""FastAPI service (requires the ``api`` extra).

Endpoints are thin: the pipeline object is built once at startup so model
weights are loaded a single time and shared across requests.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from vqa import __version__
from vqa.pipeline import AnalysisPipeline

app = FastAPI(
    title="Vision QA Pipeline",
    version=__version__,
    description="Automated product-photo quality assessment with pgvector similarity search.",
)


@lru_cache(maxsize=1)
def get_pipeline() -> AnalysisPipeline:
    return AnalysisPipeline()


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "version": __version__,
        "encoder": pipeline.encoder.name,
        "embedding_dim": pipeline.encoder.dim,
        "text_search": pipeline.encoder.supports_text,
        "vlm_provider": pipeline.critic.provider,
    }


@app.post("/analyze", summary="Score an uploaded image without storing it")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        analysis = get_pipeline().analyse_bytes(raw, filename=file.filename or "upload")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}") from exc
    return analysis.to_dict()


@app.post("/index", summary="Score an uploaded image and store it in pgvector")
async def index(file: UploadFile = File(...), sku: str | None = None) -> dict[str, Any]:
    from dataclasses import replace

    from vqa import repository

    raw = await file.read()
    analysis = get_pipeline().analyse_bytes(raw, filename=file.filename or "upload")
    if sku:
        analysis.record = replace(analysis.record, sku=sku)
    image_id = repository.upsert_analysis(analysis)
    return {"image_id": image_id, **analysis.to_dict()}


@app.get("/images/{image_id}")
def get_image(image_id: int) -> dict[str, Any]:
    from vqa import repository

    row = repository.get_image(image_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return dict(row)


@app.get("/images")
def list_images(
    limit: int = Query(50, le=500), offset: int = 0, verdict: str | None = None
) -> list[dict[str, Any]]:
    from vqa import repository

    return [dict(r) for r in repository.list_images(limit=limit, offset=offset, verdict=verdict)]


@app.get("/search/similar/{image_id}", summary="Visually similar images (cosine, HNSW)")
def similar(
    image_id: int, k: int = Query(10, le=100), verdict: str | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    from vqa import repository

    embedding = repository.get_embedding(image_id)
    if embedding is None:
        raise HTTPException(status_code=404, detail="No embedding for that image")
    hits = repository.search_similar(embedding, limit=k, exclude_id=image_id,
                                     verdict=verdict, min_score=min_score)
    return [h.to_dict() for h in hits]


@app.post("/search/by-image")
async def search_by_image(file: UploadFile = File(...), k: int = Query(10, le=100),
                          ) -> list[dict[str, Any]]:
    from vqa import repository

    raw = await file.read()
    analysis = get_pipeline().analyse_bytes(raw, filename=file.filename or "query")
    return [h.to_dict() for h in repository.search_similar(analysis.embedding, limit=k)]


@app.get("/search/by-text")
def search_by_text(q: str, k: int = Query(10, le=100)) -> list[dict[str, Any]]:
    from vqa import repository

    try:
        vector = get_pipeline().embed_text(q)
    except ValueError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return [h.to_dict() for h in repository.search_similar(vector, limit=k)]


@app.get("/stats")
def stats() -> dict[str, Any]:
    from vqa import repository

    return repository.stats()
