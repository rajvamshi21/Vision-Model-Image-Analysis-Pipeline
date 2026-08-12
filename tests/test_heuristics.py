import numpy as np
from factories import blurred, darkened, metrics_for, noisy, product_image

from vqa.quality.heuristics import edge_energy, luminance, otsu_threshold
from vqa.imageio import to_array


def test_sharp_image_scores_higher_than_blurred():
    base = product_image()
    assert metrics_for(base).sharpness > metrics_for(blurred(base)).sharpness + 0.4


def test_blur_drives_sharpness_towards_zero():
    assert metrics_for(blurred(product_image(), radius=8.0)).sharpness < 0.2


def test_dark_image_loses_exposure_but_keeps_sharpness():
    base = product_image()
    dark = metrics_for(darkened(base))
    # Contrast-normalised sharpness must not collapse just because it is dark.
    assert dark.exposure < 0.4
    assert dark.sharpness > 0.5


def test_noise_lowers_the_noise_subscore():
    base = product_image()
    assert metrics_for(noisy(base)).noise < metrics_for(base).noise - 0.3


def test_white_background_is_not_treated_as_overexposure():
    m = metrics_for(product_image())
    assert m.exposure == 1.0            # subject sits inside the plateau
    assert m.background > 0.8
    assert m.mean_luminance > 0.8       # ...even though the frame is mostly white


def test_subject_metrics_ignore_the_sweep():
    m = metrics_for(product_image(color=(30, 30, 35)))
    assert m.subject_luminance < 0.4
    assert 0.05 < m.subject_coverage < 0.5


def test_edge_energy_survives_a_large_flat_background():
    frame = np.full((400, 400), 0.95, dtype=np.float32)
    frame[190:210, 190:210] = 0.1       # small, very sharp square
    mask = frame < 0.5
    assert edge_energy(frame, mask) > edge_energy(frame, None)


def test_otsu_separates_a_bimodal_distribution():
    rng = np.random.default_rng(0)
    background = np.clip(rng.normal(0.10, 0.03, 800), 0, None)
    foreground = np.clip(rng.normal(0.70, 0.05, 400), 0, None)
    threshold = otsu_threshold(np.concatenate([background, foreground]))
    assert background.mean() < threshold < foreground.mean()
    assert (background > threshold).mean() < 0.05
    assert (foreground > threshold).mean() > 0.95


def test_luminance_matches_rec709():
    white = np.ones((2, 2, 3), dtype=np.float32)
    assert np.allclose(luminance(white), 1.0)
    # One pure primary at a time must recover its Rec.709 coefficient.
    # Note: .item(), not float() -- NumPy >= 2.3 rejects float() on ndim > 0.
    for channel, expected in enumerate((0.2126, 0.7152, 0.0722)):
        pixel = np.zeros((1, 1, 3), dtype=np.float32)
        pixel[..., channel] = 1.0
        assert abs(luminance(pixel).item() - expected) < 1e-5


def test_resolution_subscore_tracks_the_short_side():
    image = product_image(size=300)
    small = metrics_for(image)
    big = metrics_for(product_image(size=1200))
    assert small.resolution < big.resolution
    assert to_array(image).shape == (300, 300, 3)
