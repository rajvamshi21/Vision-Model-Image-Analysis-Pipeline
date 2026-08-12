-- Schema for the vision QA pipeline.
-- Applied automatically by docker-entrypoint-initdb.d, or via `vqa init-db`.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per unique image (deduplicated on SHA-256 of the file bytes).
CREATE TABLE IF NOT EXISTS images (
    id            BIGSERIAL PRIMARY KEY,
    content_hash  TEXT        NOT NULL UNIQUE,
    source_uri    TEXT        NOT NULL,
    filename      TEXT,
    sku           TEXT,
    width         INTEGER     NOT NULL,
    height        INTEGER     NOT NULL,
    size_bytes    BIGINT,
    format        TEXT,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS images_sku_idx      ON images (sku);
CREATE INDEX IF NOT EXISTS images_metadata_idx ON images USING gin (metadata jsonb_path_ops);

-- Embeddings live in their own table so the vector column can be re-built
-- (different model / dimensionality) without touching image metadata.
CREATE TABLE IF NOT EXISTS image_embeddings (
    image_id   BIGINT      PRIMARY KEY REFERENCES images (id) ON DELETE CASCADE,
    model      TEXT        NOT NULL,
    dim        INTEGER     NOT NULL,
    embedding  vector(512) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cosine distance (<=>) over L2-normalised vectors == 1 - cosine similarity.
CREATE INDEX IF NOT EXISTS image_embeddings_hnsw_idx
    ON image_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Quality verdicts, versioned so scoring changes are auditable.
CREATE TABLE IF NOT EXISTS quality_reports (
    id               BIGSERIAL PRIMARY KEY,
    image_id         BIGINT      NOT NULL REFERENCES images (id) ON DELETE CASCADE,
    score            REAL        NOT NULL,
    verdict          TEXT        NOT NULL CHECK (verdict IN ('pass', 'review', 'fail')),
    technical        JSONB       NOT NULL,
    semantic         JSONB,
    vlm              JSONB,
    issues           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    pipeline_version TEXT        NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (image_id, pipeline_version)
);

CREATE INDEX IF NOT EXISTS quality_reports_verdict_idx ON quality_reports (verdict, score DESC);
CREATE INDEX IF NOT EXISTS quality_reports_image_idx   ON quality_reports (image_id);
