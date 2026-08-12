#!/usr/bin/env python3
"""Streamlit review queue: browse scored images and search by visual similarity.

Runs in two modes:
  * in-memory  -- analyses a folder on the fly, no database needed
  * pgvector   -- reads what ``vqa ingest`` already indexed

    streamlit run demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa.imageio import iter_image_paths  # noqa: E402
from vqa.pipeline import AnalysisPipeline  # noqa: E402
from vqa.quality.vlm import NullCritic  # noqa: E402

VERDICT_STYLE = {"pass": ("#1a7f37", "PASS"), "review": ("#9a6700", "REVIEW"),
                 "fail": ("#cf222e", "FAIL")}

st.set_page_config(page_title="Vision QA Pipeline", layout="wide")


@st.cache_resource
def get_pipeline() -> AnalysisPipeline:
    return AnalysisPipeline(critic=NullCritic())


@st.cache_data(show_spinner="Analysing images...")
def analyse_folder(folder: str, limit: int):
    pipeline = get_pipeline()
    paths = iter_image_paths(folder)[:limit]
    analyses = []
    for start in range(0, len(paths), pipeline.settings.batch_size):
        analyses.extend(pipeline.analyse_paths(paths[start : start + pipeline.settings.batch_size]))
    return (
        [a.record.source_uri for a in analyses],
        [a.report.to_dict() for a in analyses],
        np.stack([a.embedding for a in analyses]) if analyses else np.zeros((0, 512)),
    )


def verdict_badge(verdict: str) -> str:
    colour, label = VERDICT_STYLE.get(verdict, ("#57606a", verdict.upper()))
    return (f"<span style='background:{colour};color:white;padding:2px 8px;"
            f"border-radius:10px;font-size:0.75rem;font-weight:600'>{label}</span>")


st.title("Vision QA Pipeline")
st.caption("Automated product-photo quality assessment + pgvector visual similarity search")

with st.sidebar:
    folder = st.text_input("Image folder", "data/sample")
    limit = st.slider("Max images", 12, 300, 126, step=6)
    st.divider()
    if st.button("Clear cache"):
        st.cache_data.clear()

if not Path(folder).exists():
    st.warning(f"`{folder}` does not exist. Run "
               "`python demo/generate_sample_images.py --out data/sample` first.")
    st.stop()

uris, reports, embeddings = analyse_folder(folder, limit)
if not uris:
    st.warning("No supported images found in that folder.")
    st.stop()

names = [Path(u).name for u in uris]
verdicts = [r["verdict"] for r in reports]
scores = [r["score"] for r in reports]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Images", len(uris))
col2.metric("Pass", verdicts.count("pass"))
col3.metric("Review", verdicts.count("review"))
col4.metric("Fail", verdicts.count("fail"))

queue_tab, detail_tab, search_tab = st.tabs(["Review queue", "Image detail", "Similarity search"])

with queue_tab:
    chosen = st.multiselect("Verdict", ["fail", "review", "pass"], default=["fail", "review"])
    order = sorted(range(len(uris)), key=lambda i: scores[i])
    shown = [i for i in order if verdicts[i] in chosen]
    st.write(f"{len(shown)} images, worst first")
    columns = st.columns(6)
    for slot, index in enumerate(shown[:60]):
        with columns[slot % 6]:
            st.image(uris[index], use_container_width=True)
            st.markdown(f"{verdict_badge(verdicts[index])} **{scores[index]:.0f}**",
                        unsafe_allow_html=True)
            st.caption(names[index])

with detail_tab:
    index = names.index(st.selectbox("Image", names, key="detail"))
    report = reports[index]
    left, right = st.columns([1, 1])
    with left:
        st.image(uris[index], use_container_width=True)
    with right:
        st.markdown(f"### {report['score']:.1f}/100 &nbsp; {verdict_badge(report['verdict'])}",
                    unsafe_allow_html=True)
        for key, value in report["technical"].items():
            if key in ("sharpness", "exposure", "contrast", "noise", "background",
                       "composition", "resolution"):
                st.progress(float(value), text=f"{key} {value:.2f}")
        if report["issues"]:
            st.markdown("#### Issues")
            for issue in report["issues"]:
                st.markdown(f"**{issue['code']}** ({issue['severity']}) — {issue['message']}  \n"
                            f"↳ _{issue['remedy']}_")
        else:
            st.success("No defects detected.")
    with st.expander("Raw measurements"):
        st.json(report["technical"])

with search_tab:
    index = names.index(st.selectbox("Query image", names, key="search"))
    k = st.slider("Neighbours", 3, 12, 6)
    similarity = embeddings @ embeddings[index]
    similarity[index] = -np.inf
    st.image(uris[index], width=220, caption=f"query: {names[index]}")
    st.markdown("#### Nearest neighbours (cosine)")
    columns = st.columns(k)
    for slot, neighbour in enumerate(np.argsort(-similarity)[:k]):
        with columns[slot]:
            st.image(uris[neighbour], use_container_width=True)
            st.caption(f"{similarity[neighbour]:.3f} · {names[neighbour]}")
