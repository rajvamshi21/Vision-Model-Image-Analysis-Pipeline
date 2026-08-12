#!/usr/bin/env python3
"""Score the labelled synthetic corpus and report how well the pipeline does.

No database required -- this is the reproducible number behind the README:

    python demo/generate_sample_images.py --out data/sample --count 120
    python demo/benchmark.py --data data/sample
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa.pipeline import AnalysisPipeline  # noqa: E402
from vqa.quality.vlm import NullCritic  # noqa: E402

SEVERE = {"high", "medium"}
DEGRADATIONS_PER_SKU = 9   # see demo/generate_sample_images.py


def load_manifest(data_dir: Path) -> list[dict]:
    with open(data_dir / "manifest.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(data_dir: Path) -> dict:
    rows = load_manifest(data_dir)
    pipeline = AnalysisPipeline(critic=NullCritic())
    paths = [data_dir / row["filename"] for row in rows]

    analyses = []
    for start in range(0, len(paths), pipeline.settings.batch_size):
        analyses.extend(pipeline.analyse_paths(paths[start : start + pipeline.settings.batch_size]))

    by_name = {a.record.filename: a for a in analyses}

    detected = defaultdict(lambda: {"hit": 0, "total": 0})
    verdict_matrix = {"clean_pass": 0, "clean_flagged": 0, "bad_caught": 0, "bad_missed": 0}
    false_alarms = defaultdict(int)
    scores = {"clean": [], "degraded": []}

    for row in rows:
        analysis = by_name[row["filename"]]
        report = analysis.report
        codes = {i.code for i in report.issues}
        severe_codes = {i.code for i in report.issues if i.severity in SEVERE}
        expected = [c for c in row["expected_issues"].split(";") if c]
        should_pass = row["should_pass"] == "1"

        if should_pass:
            scores["clean"].append(report.score)
            if report.verdict == "pass":
                verdict_matrix["clean_pass"] += 1
            else:
                verdict_matrix["clean_flagged"] += 1
            for code in severe_codes:
                false_alarms[code] += 1
        else:
            scores["degraded"].append(report.score)
            if report.verdict != "pass":
                verdict_matrix["bad_caught"] += 1
            else:
                verdict_matrix["bad_missed"] += 1
            for code in expected:
                detected[code]["total"] += 1
                detected[code]["hit"] += int(code in codes)

    # --- retrieval: can we find the clean sibling of a degraded shot? -----
    names = [a.record.filename for a in analyses]
    matrix = np.stack([a.embedding for a in analyses])
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)
    sku_of = {r["filename"]: r["sku"] for r in rows}
    degradation_of = {r["filename"]: r["degradation"] for r in rows}
    clean_index = {sku_of[n]: i for i, n in enumerate(names) if degradation_of[n] == "clean"}

    sibling_hits = same_sku_top1 = queries = 0
    precision_at_5 = []
    for i, name in enumerate(names):
        if degradation_of[name] == "clean":
            continue
        target = clean_index.get(sku_of[name])
        if target is None:
            continue
        order = np.argsort(-similarity[i])[:5]
        queries += 1
        sibling_hits += int(target in order)
        same_sku_top1 += int(sku_of[names[order[0]]] == sku_of[name])
        precision_at_5.append(
            sum(sku_of[names[j]] == sku_of[name] for j in order) / 5.0
        )

    n_clean = max(1, len(scores["clean"]))
    n_bad = max(1, len(scores["degraded"]))
    return {
        "images": len(rows),
        "encoder": pipeline.encoder.name,
        "per_defect_recall": {
            code: round(v["hit"] / v["total"], 3) for code, v in sorted(detected.items())
        },
        "clean_pass_rate": round(verdict_matrix["clean_pass"] / n_clean, 3),
        "defect_catch_rate": round(verdict_matrix["bad_caught"] / n_bad, 3),
        "balanced_accuracy": round(
            0.5 * (verdict_matrix["clean_pass"] / n_clean + verdict_matrix["bad_caught"] / n_bad), 3
        ),
        "false_alarms_on_clean": dict(sorted(false_alarms.items(), key=lambda kv: -kv[1])),
        "mean_score_clean": round(float(np.mean(scores["clean"])), 1),
        "mean_score_degraded": round(float(np.mean(scores["degraded"])), 1),
        "retrieval": {
            "queries": queries,
            "same_sku_precision_at_5": round(float(np.mean(precision_at_5)), 3),
            "same_sku_top1": round(same_sku_top1 / max(1, queries), 3),
            "clean_sibling_recall_at_5": round(sibling_hits / max(1, queries), 3),
            "chance_precision": round(
                (DEGRADATIONS_PER_SKU - 1) / max(1, len(names) - 1), 4
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/sample")
    parser.add_argument("--json", help="also write the metrics to this path")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not (data_dir / "manifest.csv").exists():
        print(f"No manifest in {data_dir}. Run demo/generate_sample_images.py first.",
              file=sys.stderr)
        return 1

    metrics = run(data_dir)
    print(json.dumps(metrics, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
