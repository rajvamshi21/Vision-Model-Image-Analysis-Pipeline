import numpy as np
from factories import as_jpeg_bytes, product_image

from vqa.imageio import downscale, iter_image_paths, load_image, sha256_file, to_array
from vqa.ingest import sku_from_filename
from vqa.repository import _vector_literal
from vqa.types import ImageRecord, QualityReport, TechnicalMetrics
from pathlib import Path


def test_sha256_is_stable(tmp_path):
    path = tmp_path / "a.jpg"
    path.write_bytes(as_jpeg_bytes(product_image()))
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_iter_image_paths_filters_and_sorts(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.jpg").write_bytes(as_jpeg_bytes(product_image()))
    (tmp_path / "nested" / "a.png").write_bytes(as_jpeg_bytes(product_image()))
    (tmp_path / "notes.txt").write_text("ignore me")
    found = [p.name for p in iter_image_paths(tmp_path)]
    assert found == ["b.jpg", "a.png"] or sorted(found) == ["a.png", "b.jpg"]
    assert "notes.txt" not in found
    assert iter_image_paths(tmp_path, recursive=False) == [tmp_path / "b.jpg"]


def test_load_image_flattens_transparency(tmp_path):
    from PIL import Image

    path = tmp_path / "a.png"
    Image.new("RGBA", (20, 20), (255, 0, 0, 0)).save(path)
    loaded = load_image(path)
    assert loaded.mode == "RGB"
    assert np.asarray(loaded).min() == 255      # alpha composited onto white


def test_downscale_bounds_the_longest_side():
    image = product_image(size=900)
    assert max(downscale(image, 300).size) == 300
    assert downscale(image, 5000).size == image.size


def test_to_array_is_normalised():
    arr = to_array(product_image(size=64))
    assert arr.dtype == np.float32 and arr.shape == (64, 64, 3)
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_sku_is_derived_from_the_filename_stem():
    assert sku_from_filename(Path("bottle-crimson_01.jpg")) == "bottle-crimson"
    assert sku_from_filename(Path("SKU12345.png")) == "SKU12345"


def test_vector_literal_matches_pgvector_syntax():
    literal = _vector_literal(np.array([1.0, -0.5, 0.25], dtype=np.float32))
    assert literal.startswith("[") and literal.endswith("]")
    assert literal.count(",") == 2
    assert "1.000000" in literal


def test_report_serialises_to_plain_json_types():
    import json

    metrics = TechnicalMetrics(*([10] * 2 + [0.5] * 21))
    report = QualityReport(score=90.0, verdict="pass", technical=metrics)
    payload = json.dumps(report.to_dict())
    assert '"verdict": "pass"' in payload

    record = ImageRecord("h", "uri", "f.jpg", 10, 10, 100, "JPEG")
    assert record.metadata == {}
