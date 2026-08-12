import numpy as np
from factories import as_jpeg_bytes, blurred, product_image

from vqa.config import Settings
from vqa.pipeline import AnalysisPipeline
from vqa.quality.vlm import NullCritic


def _pipeline(**kwargs):
    return AnalysisPipeline(settings=Settings(**kwargs), critic=NullCritic())


def test_analyse_bytes_returns_a_report_and_an_embedding():
    analysis = _pipeline().analyse_bytes(as_jpeg_bytes(product_image()), filename="a.jpg")
    assert analysis.report.verdict == "pass"
    assert analysis.embedding.shape == (512,)
    assert np.isclose(np.linalg.norm(analysis.embedding), 1.0, atol=1e-5)
    assert analysis.record.width == analysis.record.height == 900
    assert len(analysis.record.content_hash) == 64


def test_identical_bytes_hash_identically():
    raw = as_jpeg_bytes(product_image())
    pipeline = _pipeline()
    assert (pipeline.analyse_bytes(raw).record.content_hash
            == pipeline.analyse_bytes(raw).record.content_hash)


def test_analyse_paths_batches_files(tmp_path):
    paths = []
    for i, image in enumerate([product_image(), blurred(product_image())]):
        path = tmp_path / f"img-{i}.jpg"
        path.write_bytes(as_jpeg_bytes(image))
        paths.append(path)
    results = _pipeline().analyse_paths(paths)
    assert [r.record.filename for r in results] == ["img-0.jpg", "img-1.jpg"]
    assert results[0].report.score > results[1].report.score


def test_iter_batches_covers_every_path(tmp_path):
    paths = []
    for i in range(5):
        path = tmp_path / f"p{i}.jpg"
        path.write_bytes(as_jpeg_bytes(product_image(color=(10 * i, 90, 200))))
        paths.append(path)
    seen = [a.record.filename for batch in _pipeline(batch_size=2).iter_batches(paths)
            for a in batch]
    assert sorted(seen) == sorted(p.name for p in paths)


def test_large_images_are_downscaled_before_analysis():
    pipeline = _pipeline(max_side=256)
    prepared = pipeline._prepare_bytes(as_jpeg_bytes(product_image(size=900)), "u", "u.jpg")
    assert max(prepared.image.size) == 256
    assert prepared.record.width == 900        # metadata keeps the original size


def test_text_search_is_refused_without_a_text_tower():
    try:
        _pipeline().embed_text("a red bottle")
    except ValueError as exc:
        assert "openclip" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_to_dict_is_json_shaped():
    payload = _pipeline().analyse_bytes(as_jpeg_bytes(product_image())).to_dict()
    assert set(payload) == {"image", "report", "embedding_model", "embedding_dim"}
    assert payload["embedding_dim"] == 512
    assert payload["report"]["technical"]["sharpness"] >= 0.0
