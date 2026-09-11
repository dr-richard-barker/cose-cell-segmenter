"""Standard image-processing operations, as pure numpy functions.

No Qt/napari imports here — these are called directly by the preprocessing
panel widget, and are exercised independently in tests/test_preprocessing.py.

Convention: a grayscale image is a plain (Y, X[, Z]) array (`channel_axis=None`).
A multichannel image carries its channel axis explicitly via `channel_axis`
(commonly the last axis, e.g. (Y, X, C)) — the caller states it, we never guess.
"""

from __future__ import annotations

import numpy as np
from skimage import exposure, filters, restoration
from skimage.morphology import disk


def _apply_per_channel(image: np.ndarray, channel_axis: int, func) -> np.ndarray:
    """Run a 2D-only skimage function independently over each channel."""
    n = image.shape[channel_axis]
    out = np.empty_like(image)
    for c in range(n):
        idx = [slice(None)] * image.ndim
        idx[channel_axis] = c
        out[tuple(idx)] = func(image[tuple(idx)])
    return out


def gaussian_blur(
    image: np.ndarray, sigma: float = 1.0, channel_axis: int | None = None
) -> np.ndarray:
    """Gaussian smoothing. `sigma` is in pixels."""
    out = filters.gaussian(
        image, sigma=sigma, channel_axis=channel_axis, preserve_range=True
    )
    return out.astype(image.dtype, copy=False)


def median_blur(
    image: np.ndarray, radius: int = 2, channel_axis: int | None = None
) -> np.ndarray:
    """Median filter with a disk-shaped footprint of the given pixel radius."""
    footprint = disk(radius)
    if channel_axis is None:
        return filters.median(image, footprint)
    return _apply_per_channel(image, channel_axis, lambda plane: filters.median(plane, footprint))


def clahe(
    image: np.ndarray,
    clip_limit: float = 0.01,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    Returns a float64 array in [0, 1] (as `skimage.exposure.equalize_adapthist`
    always does), regardless of the input dtype — rescale for display/export
    as needed.
    """
    if channel_axis is None or channel_axis in (image.ndim - 1, -1):
        return exposure.equalize_adapthist(image, clip_limit=clip_limit)
    moved = np.moveaxis(image, channel_axis, -1)
    result = exposure.equalize_adapthist(moved, clip_limit=clip_limit)
    return np.moveaxis(result, -1, channel_axis)


def rolling_ball_bg_subtract(
    image: np.ndarray, radius: int = 50, channel_axis: int | None = None
) -> np.ndarray:
    """Estimate and subtract an uneven background via the rolling-ball algorithm.

    Returns the same dtype as the input, clipped to a non-negative range.
    """
    if channel_axis is None:
        background = restoration.rolling_ball(image, radius=radius)
    else:
        background = _apply_per_channel(
            image, channel_axis, lambda plane: restoration.rolling_ball(plane, radius=radius)
        )
    subtracted = image.astype(np.float64) - background.astype(np.float64)
    subtracted = np.clip(subtracted, 0, None)
    return subtracted.astype(image.dtype, copy=False)
