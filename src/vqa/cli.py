"""Command line interface: ``vqa <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vqa import __version__
from vqa.config import get_settings

BAR_WIDTH = 24
COLOURS = {"pass": "\033[32m", "review": "\033[33m", "fail": "\033[31m"}
RESET = "\033[0m"


def _colour(text: str, verdict: str, enabled: bool) -> str:
    if not enabled or verdict not in COLOURS:
        return text
    return f"{COLOURS[verdict]}{text}{RESET}"


def _bar(fraction: float) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * BAR_WIDTH))
    return "#" * filled + "." * (BAR_WIDTH - filled)


def _print_report(analysis, use_colour: bool) -> None:
    report = analysis.report
    header = f"{analysis.record.filename}  {report.score:>5.1f}/100  {report.verdict.upper()}"
    print(_colour(header, report.verdict, use_colour))
    for name, value in report.technical.subscores().items():
        print(f"    {name:<12} {_bar(value)} {value:.2f}")
    if report.semantic:
        top = sorted(report.semantic.scores.items(), key=lambda kv: -kv[1])[:4]
        print("    zero-shot   " + ", ".join(f"{k}={v:.2f}" for k, v in top))
    if report.vlm and report.vlm.caption:
        print(f"    vlm         {report.vlm.caption}")
    for issue in report.issues:
        print(f"    [{issue.severity:<6}] {issue.code}: {issue.message}")
        print(f"             -> {issue.remedy}")
    print()


def cmd_init_db(args) -> int:
    from vqa.db import init_schema

    init_schema()
    print(f"Schema applied to {get_settings().database_url}")
    return 0


def cmd_analyze(args) -> int:
    from vqa.imageio import iter_image_paths
    from vqa.pipeline import AnalysisPipeline

    paths: list[Path] = []
    for target in args.paths:
        paths.extend(iter_image_paths(target, recursive=not args.no_recursive))
    if not paths:
        print("No supported images found.", file=sys.stderr)
        return 1

    pipeline = AnalysisPipeline()
    results = []
    use_colour = sys.stdout.isatty() and not args.json
    for batch in pipeline.iter_batches(paths):
        for analysis in batch:
            results.append(analysis.to_dict())
            if not args.json:
                _print_report(analysis, use_colour)

    if args.json:
        payload = json.dumps(results, indent=2)
        if args.json == "-":
            print(payload)
        else:
            Path(args.json).write_text(payload, encoding="utf-8")
            print(f"Wrote {len(results)} reports to {args.json}")
    else:
        counts: dict[str, int] = {}
        for r in results:
            counts[r["report"]["verdict"]] = counts.get(r["report"]["verdict"], 0) + 1
        plural = "" if len(results) == 1 else "s"
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{len(results)} image{plural}: {summary}")
    return 0


def cmd_ingest(args) -> int:
    from vqa.ingest import ingest_directory

    result = ingest_directory(
        args.path,
        recursive=not args.no_recursive,
        sku_from_name=args.sku_from_name,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 1 if result.failed and not result.ingested else 0


def cmd_search(args) -> int:
    from vqa import repository
    from vqa.pipeline import AnalysisPipeline

    if args.image:
        pipeline = AnalysisPipeline()
        analysis = pipeline.analyse_bytes(Path(args.image).read_bytes(), Path(args.image).name)
        vector = analysis.embedding
    elif args.text:
        vector = AnalysisPipeline().embed_text(args.text)
    elif args.id is not None:
        vector = repository.get_embedding(args.id)
        if vector is None:
            print(f"No embedding stored for image id {args.id}", file=sys.stderr)
            return 1
    else:
        print("Provide --image, --text or --id", file=sys.stderr)
        return 1

    hits = repository.search_similar(
        vector, limit=args.k, exclude_id=args.id, verdict=args.verdict, min_score=args.min_score
    )
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], indent=2))
        return 0
    print(f"{'sim':>6}  {'score':>6}  {'verdict':<7}  file")
    for hit in hits:
        score = "-" if hit.score is None else f"{hit.score:6.1f}"
        print(f"{hit.similarity:6.3f}  {score}  {(hit.verdict or '-'):<7}  {hit.filename}")
    return 0


def cmd_stats(args) -> int:
    from vqa import repository

    print(json.dumps(repository.stats(), indent=2))
    return 0


def cmd_config(args) -> int:
    from dataclasses import asdict

    settings = asdict(get_settings())
    settings["database_url"] = settings["database_url"].split("@")[-1]  # never print credentials
    print(json.dumps(settings, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("vqa.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vqa", description="Vision-model image analysis pipeline"
    )
    parser.add_argument("--version", action="version", version=f"vqa {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="create schema, extension and indexes")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("analyze", help="score images and print reports (no database)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--json", nargs="?", const="-", help="write JSON to a path, or '-' for stdout")
    p.add_argument("--no-recursive", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("ingest", help="analyse a directory and index it in pgvector")
    p.add_argument("path")
    p.add_argument("--sku-from-name", action="store_true", help="derive SKU from the filename stem")
    p.add_argument("--force", action="store_true", help="re-process already-indexed images")
    p.add_argument("--dry-run", action="store_true", help="analyse without writing to the database")
    p.add_argument("--no-recursive", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="nearest-neighbour search over the indexed corpus")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--image", help="query by an image file")
    group.add_argument("--text", help="query by text (requires a CLIP encoder)")
    group.add_argument("--id", type=int, help="query by an indexed image id")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--verdict", choices=["pass", "review", "fail"])
    p.add_argument("--min-score", type=float)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", help="corpus-level quality summary")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("config", help="show the resolved configuration")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("serve", help="run the REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger("vqa").error("%s", exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
