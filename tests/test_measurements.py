import numpy as np

from cose_cell_segmenter.core import measurements as meas


def _two_square_mask():
    """A 10x10 label image: a 3x3 square (label 1, area 9) and a 2x2 square
    (label 2, area 4), non-touching, with a known answer for `area`.
    """
    masks = np.zeros((10, 10), dtype=np.int32)
    masks[1:4, 1:4] = 1  # 3x3 = 9 px
    masks[6:8, 6:8] = 2  # 2x2 = 4 px
    return masks


def test_measure_known_areas():
    df = meas.measure(_two_square_mask())
    assert sorted(df["label"].tolist()) == [1, 2]
    areas = dict(zip(df["label"], df["area"]))
    assert areas[1] == 9
    assert areas[2] == 4


def test_measure_empty_mask_returns_empty_frame_with_columns():
    masks = np.zeros((10, 10), dtype=np.int32)
    df = meas.measure(masks)
    assert len(df) == 0
    assert "label" in df.columns
    assert "area" in df.columns


def test_measure_includes_intensity_columns_when_given():
    masks = _two_square_mask()
    intensity = np.zeros((10, 10), dtype=np.float32)
    intensity[1:4, 1:4] = 100.0  # label 1 region, uniform intensity 100
    df = meas.measure(masks, intensity_image=intensity)
    row = df[df["label"] == 1].iloc[0]
    assert row["intensity_mean"] == 100.0
    assert row["intensity_max"] == 100.0


def test_measure_pixel_size_calibration():
    df = meas.measure(_two_square_mask(), pixel_size=0.5)  # e.g. 0.5 microns/px
    row = df[df["label"] == 1].iloc[0]
    assert row["area"] == 9
    assert row["area_calibrated"] == 9 * 0.25  # pixel_size**2
