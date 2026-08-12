"""Directory -> pgvector ingestion with content-hash deduplication."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from vqa import repository
from vqa.imageio import iter_image_paths, sha256_file
from vqa.pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)

SKU_PATTERN = re.compile(r"^([A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*?)(?:[-_](?:\d+|v\d+))?$")


@dataclass
class IngestResult:
    scanned: int = 0
    skipped: int = 0
    ingested: int = 0
    failed: int = 0
    verdicts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned, "skipped": self.skipped, "ingested": self.ingested,
            "failed": self.failed, "verdicts": self.verdicts, "errors": self.errors[:10],
        }


def sku_from_filename(path: Path) -> str | None:
    match = SKU_PATTERN.match(path.stem)
    return match.group(1) if match else None


def ingest_directory(
    root: str | Path,
    pipeline: AnalysisPipeline | None = None,
    recursive: bool = True,
    sku_from_name: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> IngestResult:
    pipeline = pipeline or AnalysisPipeline()
    paths = iter_image_paths(root, recursive=recursive)
    result = IngestResult(scanned=len(paths))
    if not paths:
        return result

    if not force and not dry_run:
        # One round-trip instead of one query per file.
        hashes = {p: sha256_file(p) for p in paths}
        known = repository.existing_hashes(list(hashes.values()))
        pending = [p for p in paths if hashes[p] not in known]
        result.skipped = len(paths) - len(pending)
        paths = pending

    skus = [sku_from_filename(p) if sku_from_name else None for p in paths]

    for batch_paths, batch_skus in _chunks(paths, skus, pipeline.settings.batch_size):
        try:
            analyses = pipeline.analyse_paths(batch_paths, batch_skus)
        except Exception as exc:
            result.failed += len(batch_paths)
            result.errors.append(f"batch starting {batch_paths[0].name}: {exc}")
            logger.exception("Batch failed")
            continue

        for analysis in analyses:
            verdict = analysis.report.verdict
            result.verdicts[verdict] = result.verdicts.get(verdict, 0) + 1
            if dry_run:
                result.ingested += 1
                continue
            try:
                repository.upsert_analysis(analysis)
                result.ingested += 1
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{analysis.record.filename}: {exc}")
                logger.exception("Failed to store %s", analysis.record.filename)
    return result


def _chunks(paths, skus, size):
    size = max(1, size)
    for start in range(0, len(paths), size):
        yield paths[start : start + size], skus[start : start + size]
