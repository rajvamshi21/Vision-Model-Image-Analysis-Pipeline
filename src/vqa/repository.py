"""Persistence for images, embeddings and quality reports.

All vector maths happens in Postgres via pgvector's cosine operator (``<=>``),
so retrieval stays correct as the corpus grows past what fits in memory.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vqa.db import connection
from vqa.types import ImageAnalysis, SearchHit

SIMILAR_SQL = """
WITH probe AS (SELECT %(vector)s::vector AS v)
SELECT i.id,
       i.source_uri,
       i.filename,
       i.sku,
       1 - (e.embedding <=> probe.v) AS similarity,
       q.score,
       q.verdict
FROM image_embeddings e
CROSS JOIN probe
JOIN images i ON i.id = e.image_id
LEFT JOIN LATERAL (
    SELECT score, verdict FROM quality_reports r
    WHERE r.image_id = i.id ORDER BY r.created_at DESC LIMIT 1
) q ON TRUE
WHERE (%(exclude_id)s::bigint IS NULL OR i.id <> %(exclude_id)s::bigint)
  AND (%(verdict)s::text IS NULL OR q.verdict = %(verdict)s::text)
  AND (%(min_score)s::real IS NULL OR q.score >= %(min_score)s::real)
ORDER BY e.embedding <=> probe.v
LIMIT %(limit)s
"""


def _vector_literal(embedding: np.ndarray) -> str:
    """pgvector accepts a '[a,b,c]' text literal -- avoids a hard pgvector-python dep."""
    values = np.asarray(embedding, dtype=np.float32).ravel().tolist()
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def upsert_analysis(analysis: ImageAnalysis) -> int:
    """Insert (or refresh) one image, its embedding and its latest report."""
    record = analysis.record
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO images (content_hash, source_uri, filename, sku,
                                    width, height, size_bytes, format, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (content_hash) DO UPDATE
                    SET source_uri = EXCLUDED.source_uri,
                        filename   = EXCLUDED.filename,
                        sku        = COALESCE(EXCLUDED.sku, images.sku),
                        metadata   = images.metadata || EXCLUDED.metadata
                RETURNING id
                """,
                (record.content_hash, record.source_uri, record.filename, record.sku,
                 record.width, record.height, record.size_bytes, record.format,
                 json.dumps(record.metadata)),
            )
            image_id = int(cur.fetchone()["id"])

            if analysis.embedding is not None:
                cur.execute(
                    """
                    INSERT INTO image_embeddings (image_id, model, dim, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (image_id) DO UPDATE
                        SET model = EXCLUDED.model,
                            dim = EXCLUDED.dim,
                            embedding = EXCLUDED.embedding,
                            created_at = now()
                    """,
                    (image_id, analysis.embedding_model, int(analysis.embedding.shape[-1]),
                     _vector_literal(analysis.embedding)),
                )

            report = analysis.report
            payload = report.to_dict()
            cur.execute(
                """
                INSERT INTO quality_reports (image_id, score, verdict, technical,
                                             semantic, vlm, issues, pipeline_version)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
                ON CONFLICT (image_id, pipeline_version) DO UPDATE
                    SET score = EXCLUDED.score,
                        verdict = EXCLUDED.verdict,
                        technical = EXCLUDED.technical,
                        semantic = EXCLUDED.semantic,
                        vlm = EXCLUDED.vlm,
                        issues = EXCLUDED.issues,
                        created_at = now()
                """,
                (image_id, report.score, report.verdict,
                 json.dumps(payload["technical"]),
                 json.dumps(payload["semantic"]) if payload["semantic"] else None,
                 json.dumps(payload["vlm"]) if payload["vlm"] else None,
                 json.dumps(payload["issues"]), report.pipeline_version),
            )
        conn.commit()
    return image_id


def existing_hashes(hashes: list[str]) -> set[str]:
    if not hashes:
        return set()
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT content_hash FROM images WHERE content_hash = ANY(%s)", (hashes,))
        return {row["content_hash"] for row in cur.fetchall()}


def search_similar(
    embedding: np.ndarray,
    limit: int = 10,
    exclude_id: int | None = None,
    verdict: str | None = None,
    min_score: float | None = None,
) -> list[SearchHit]:
    params = {
        "vector": _vector_literal(embedding),
        "limit": limit,
        "exclude_id": exclude_id,
        "verdict": verdict,
        "min_score": min_score,
    }
    with connection() as conn, conn.cursor() as cur:
        cur.execute(SIMILAR_SQL, params)
        rows = cur.fetchall()
    return [
        SearchHit(
            image_id=row["id"], source_uri=row["source_uri"], filename=row["filename"],
            sku=row["sku"], similarity=round(float(row["similarity"]), 4),
            score=None if row["score"] is None else float(row["score"]),
            verdict=row["verdict"],
        )
        for row in rows
    ]


def get_embedding(image_id: int) -> np.ndarray | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT embedding FROM image_embeddings WHERE image_id = %s", (image_id,))
        row = cur.fetchone()
    if not row:
        return None
    value = row["embedding"]
    if isinstance(value, str):
        return np.array(json.loads(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def get_image(image_id: int) -> dict[str, Any] | None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT i.*, q.score, q.verdict, q.issues, q.technical, q.semantic, q.vlm
            FROM images i
            LEFT JOIN LATERAL (
                SELECT * FROM quality_reports r
                WHERE r.image_id = i.id ORDER BY r.created_at DESC LIMIT 1
            ) q ON TRUE
            WHERE i.id = %s
            """,
            (image_id,),
        )
        return cur.fetchone()


def list_images(limit: int = 50, offset: int = 0, verdict: str | None = None) -> list[dict]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT i.id, i.filename, i.source_uri, i.sku, i.width, i.height,
                   q.score, q.verdict
            FROM images i
            LEFT JOIN LATERAL (
                SELECT score, verdict FROM quality_reports r
                WHERE r.image_id = i.id ORDER BY r.created_at DESC LIMIT 1
            ) q ON TRUE
            WHERE (%s::text IS NULL OR q.verdict = %s::text)
            ORDER BY q.score ASC NULLS LAST, i.id
            LIMIT %s OFFSET %s
            """,
            (verdict, verdict, limit, offset),
        )
        return cur.fetchall()


def stats() -> dict[str, Any]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM images")
        total = int(cur.fetchone()["n"])
        cur.execute(
            """
            SELECT verdict, count(*) AS n, avg(score)::numeric(5,1) AS avg_score
            FROM quality_reports GROUP BY verdict ORDER BY verdict
            """
        )
        verdicts = {r["verdict"]: {"count": int(r["n"]), "avg_score": float(r["avg_score"])}
                    for r in cur.fetchall()}
        cur.execute(
            """
            SELECT issue ->> 'code' AS code, count(*) AS n
            FROM quality_reports, jsonb_array_elements(issues) AS issue
            GROUP BY 1 ORDER BY n DESC LIMIT 10
            """
        )
        top_issues = [{"code": r["code"], "count": int(r["n"])} for r in cur.fetchall()]
    return {"images": total, "verdicts": verdicts, "top_issues": top_issues}
