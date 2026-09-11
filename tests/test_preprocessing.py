import numpy as np

from cose_cell_segmenter.core import preprocessing as pp


def test_gaussian_blur_preserves_flat_image():
    image = np.full((20, 20), 100, dtype=np.uint8)
    out = pp.gaussian_blur(image, sigma=2.0)
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    # a perfectly flat image is its own blur (mode='nearest' keeps edges flat too)
    assert np.allclose(out, 100, atol=1)


def test_gaussian_blur_multichannel():
    image = np.full((20, 20, 3), 50, dtype=np.uint8)
    out = pp.gaussian_blur(image, sigma=1.5, channel_axis=-1)
    assert out.shape == image.shape
    assert np.allclose(out, 50, atol=1)


def test_median_blur_preserves_flat_image():
    image = np.full((20, 20), 77, dtype=np.uint8)
    out = pp.median_blur(image, radius=2)
    assert out.shape == image.shape
    assert np.array_equal(out, image)


def test_median_blur_removes_salt_pepper_spike():
    image = np.full((15, 15), 10, dtype=np.uint8)
    image[7, 7] = 255  # single-pixel outlier
    out = pp.median_blur(image, radius=2)
    # a lone spike is removed by a median filter with a radius-2 footprint
    assert out[7, 7] != 255
    assert out[7, 7] == 10


def test_median_blur_multichannel_matches_per_channel_loop():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(12, 12, 2), dtype=np.uint8)
    out = pp.median_blur(image, radius=1, channel_axis=-1)
    assert out.shape == image.shape
    # cross-check against running the single-channel path manually per channel
    expected_c0 = pp.median_blur(image[..., 0], radius=1)
    expected_c1 = pp.median_blur(image[..., 1], radius=1)
    assert np.array_equal(out[..., 0], expected_c0)
    assert np.array_equal(out[..., 1], expected_c1)


def test_clahe_output_in_unit_range():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)
    out = pp.clahe(image, clip_limit=0.02)
    assert out.shape == image.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_clahe_multichannel_roundtrips_axis():
    rng = np.random.default_rng(2)
    # channel-first array (C, Y, X) — clahe internally expects channel-last
    image = rng.integers(0, 255, size=(3, 32, 32), dtype=np.uint8)
    out = pp.clahe(image, channel_axis=0)
    assert out.shape == image.shape


def test_rolling_ball_flat_image_has_near_zero_background():
    image = np.full((40, 40), 50, dtype=np.uint16)
    out = pp.rolling_ball_bg_subtract(image, radius=20)
    # a perfectly flat image IS the background -> subtracting it leaves ~0
    assert out.max() <= 2


def test_rolling_ball_preserves_a_bright_spike():
    image = np.full((60, 60), 20, dtype=np.uint16)
    image[30, 30] = 220
    out = pp.rolling_ball_bg_subtract(image, radius=15)
    # the spike should survive background subtraction, clearly above the
    # (now near-zero) surrounding background
    assert out[30, 30] > out[10, 10] + 50
