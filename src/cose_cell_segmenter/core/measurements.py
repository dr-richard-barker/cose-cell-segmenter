"""Per-object measurements from a label mask, via skimage.measure.regionprops_table.

Cellpose itself only returns label masks — it does not compute measurements.
This wraps the ecosystem-standard next step (the same approach used by the
napari-serialcellpose plugin) and returns a pandas DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

#: A sensible default column set for cell/nucleus/spore counting work.
DEFAULT_PROPERTIES = (
    "label",
    "area",
    "centroid",
    "perimeter",
    "eccentricity",
    "solidity",
    "equivalent_diameter_area",
)

#: Extra columns available only when an intensity image is supplied.
INTENSITY_PROPERTIES = ("intensity_mean", "intensity_min", "intensity_max")


def measure(
    masks: np.ndarray,
    intensity_image: np.ndarray | None = None,
    properties: tuple[str, ...] | None = None,
    pixel_size: float | None = None,
) -> pd.DataFrame:
    """Compute a per-object measurements table.

    `masks` is a labeled integer array (0 = background), as returned by
    `core.segmentation.run`. If `intensity_image` is given, intensity-based
    columns (mean/min/max) are included automatically.

    `pixel_size`, if given (physical units per pixel, e.g. microns/px), scales
    `area` into `area_calibrated` (pixel_size**2) and
    `equivalent_diameter_area` into `equivalent_diameter_calibrated`
    (pixel_size) as extra columns — the raw pixel-unit columns are kept too.
    """
    props = list(properties) if properties else list(DEFAULT_PROPERTIES)
    if intensity_image is not None:
        props += [p for p in INTENSITY_PROPERTIES if p not in props]

    if not masks.any():
        return pd.DataFrame(columns=props)

    table = regionprops_table(
        masks,
        intensity_image=intensity_image,
        properties=tuple(props),
    )
    df = pd.DataFrame(table)

    if pixel_size:
        if "area" in df.columns:
            df["area_calibrated"] = df["area"] * (pixel_size**2)
        if "equivalent_diameter_area" in df.columns:
            df["equivalent_diameter_calibrated"] = df["equivalent_diameter_area"] * pixel_size

    return df
