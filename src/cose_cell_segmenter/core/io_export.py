"""Thin wrappers over cellpose.io + pandas CSV export, for a single image/mask
pair (our GUI works on one image at a time, unlike cellpose's own batch-CLI
functions which expect parallel lists).

No Qt/napari imports here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from cellpose import io as cp_io
from skimage.io import imsave


def save_masks_array(
    masks: np.ndarray,
    base_path: str | Path,
    png: bool = True,
    tif: bool = False,
) -> list[Path]:
    """Save the labeled mask array directly, with no dependency on Cellpose's
    per-run flow data — safe to use on the full accumulated "masks" canvas
    (which may combine several segmented regions with no single associated
    flow field). Written as uint16 (supports up to 65535 objects).

    This intentionally bypasses `cellpose.io.save_masks`, which requires real
    flow data to render its annotated preview figure — use `save_project`
    below (scoped to a single segmentation run) if you need that.
    """
    masks16 = masks.astype(np.uint16)
    written = []
    if png:
        path = Path(f"{base_path}_cp_masks.png")
        imsave(path, masks16, check_contrast=False)
        written.append(path)
    if tif:
        path = Path(f"{base_path}_cp_masks.tif")
        tifffile.imwrite(path, masks16)
        written.append(path)
    return written


def save_rois(masks: np.ndarray, base_path: str | Path) -> Path | None:
    """Export mask outlines as an ImageJ/Fiji-importable ROI bundle (.zip).

    `cellpose.io.save_rois` appends its own `_rois.zip` suffix internally
    (after stripping whatever extension it's given via `os.path.splitext`) —
    so we pass the bare base path, not a pre-suffixed one, to avoid ending up
    with `<base>_rois_rois.zip`.

    Returns None (and writes nothing) if `masks` is empty — cellpose's own
    function silently no-ops in that case rather than raising.
    """
    if masks.max() == 0:
        return None
    out_path = Path(f"{base_path}_rois.zip")
    cp_io.save_rois(masks, str(base_path))
    return out_path


def save_project(
    image: np.ndarray, masks: np.ndarray, flows: list, base_path: str | Path
) -> Path:
    """Save cellpose's native `_seg.npy` project file — reopenable in this app
    or in Cellpose's official GUI.

    `flows` must be the real `flows` list returned alongside `masks` by
    `core.segmentation.SegmentationEngine.run` for THIS SAME image/masks pair
    — cellpose's own `io.py` dereferences `flows[0]`/`flows[1]`/`flows[2]`
    unconditionally, so this only makes sense for a single segmentation run
    (image crop + its own masks + its own flows), not an accumulated
    multi-region canvas. Use `save_masks_array` for that instead.
    """
    cp_io.masks_flows_to_seg([image], [masks], [flows], [str(base_path)])
    return Path(f"{base_path}_seg.npy")


def save_measurements_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    df.to_csv(path, index=False)
    return path
