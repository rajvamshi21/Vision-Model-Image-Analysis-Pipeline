"""Postgres connection handling (requires the ``db`` extra)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

from vqa.config import Settings, get_settings

logger = logging.getLogger(__name__)

SCHEMA_CANDIDATES = (
    # running from a checkout: src/vqa/db.py -> <repo>/db/init/001_schema.sql
    Path(__file__).resolve().parents[2] / "db" / "init" / "001_schema.sql",
    # running from an installed package (Docker image, site-packages)
    Path.cwd() / "db" / "init" / "001_schema.sql",
    Path("/app/db/init/001_schema.sql"),
)


def find_schema() -> Path:
    """Locate the DDL from a checkout, the Docker image, or ``VQA_SCHEMA_PATH``."""
    override = os.getenv("VQA_SCHEMA_PATH", "").strip()
    candidates = (Path(override),) + SCHEMA_CANDIDATES if override else SCHEMA_CANDIDATES
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find 001_schema.sql. Run from the repository root or set VQA_SCHEMA_PATH."
    )

_POOL = None


def _require_psycopg():
    try:
        import psycopg  # noqa: F401
        from psycopg_pool import ConnectionPool  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Postgres support needs the db extra: pip install 'vqa[db]'"
        ) from exc


def get_pool(settings: Settings | None = None):
    """Lazily build a process-wide connection pool with pgvector registered."""
    global _POOL
    if _POOL is not None:
        return _POOL

    _require_psycopg()
    from pgvector.psycopg import register_vector
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    settings = settings or get_settings()

    def configure(conn):
        conn.row_factory = dict_row
        try:
            register_vector(conn)
        except Exception:  # extension not created yet (first `init-db` run)
            logger.debug("pgvector types not registered yet")

    _POOL = ConnectionPool(settings.database_url, min_size=1, max_size=8,
                           configure=configure, open=True)
    return _POOL


@contextmanager
def connection(settings: Settings | None = None):
    pool = get_pool(settings)
    with pool.connection() as conn:
        yield conn


def init_schema(settings: Settings | None = None, schema_path: Path | None = None) -> None:
    """Apply the DDL. Idempotent -- every statement is IF NOT EXISTS."""
    path = schema_path or find_schema()
    sql = path.read_text(encoding="utf-8")
    with connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    # Re-register vector types now that the extension definitely exists.
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None
    logger.info("Schema applied from %s", path)


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None
