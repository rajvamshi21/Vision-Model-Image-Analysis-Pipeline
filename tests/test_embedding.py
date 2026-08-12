import numpy as np
from factories import darkened, noisy, product_image

from vqa.config import Settings
from vqa.embedding import get_encoder
from vqa.embedding.hashing import PerceptualHashEncoder


def test_encoder_shape_and_normalisation():
    encoder = PerceptualHashEncoder()
    vectors = encoder.encode_images([product_image(), product_image(color=(40, 90, 200))])
    assert vectors.shape == (2, 512) == (2, encoder.dim)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert vectors.dtype == np.float32


def test_encoding_is_deterministic():
    encoder = PerceptualHashEncoder()
    image = product_image()
    assert np.allclose(encoder.encode_images([image]), encoder.encode_images([image]))


def test_same_product_is_closer_than_a_different_one():
    encoder = PerceptualHashEncoder()
    base = product_image(color=(200, 70, 60))
    same = noisy(base, sigma=6.0)
    other = product_image(color=(40, 90, 200), scale=0.7, offset=120)
    vectors = encoder.encode_images([base, same, other])
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_descriptor_is_robust_to_exposure_changes():
    encoder = PerceptualHashEncoder()
    base = product_image()
    vectors = encoder.encode_images([base, darkened(base, gain=0.4)])
    assert float(vectors[0] @ vectors[1]) > 0.9


def test_empty_batch_returns_empty_matrix():
    assert PerceptualHashEncoder().encode_images([]).shape == (0, 512)


def test_factory_returns_the_hash_backend_by_default():
    encoder = get_encoder(Settings(embedding_backend="hash"), cache=False)
    assert isinstance(encoder, PerceptualHashEncoder)
    assert encoder.supports_text is False


def test_unknown_backend_is_rejected():
    try:
        get_encoder(Settings(embedding_backend="nope"), cache=False)
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError")
