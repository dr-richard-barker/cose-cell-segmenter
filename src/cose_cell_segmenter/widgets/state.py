"""Shared mutable state for the dock panels, owned by the main app window.

Keeps the panels loosely coupled (each reads/writes fields here instead of
reaching into each other directly) without building a full event bus for a
single-window desktop tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cose_cell_segmenter.core.segmentation import SegmentationEngine


@dataclass
class LastRun:
    """The image crop, masks, and flows from the most recent single
    segmentation run — kept together because cellpose's own `_seg.npy`
    project export needs a self-consistent (image, masks, flows) triple, not
    the accumulated multi-region "masks" canvas.
    """

    image: np.ndarray
    masks: np.ndarray
    flows: list[Any]


@dataclass
class AppState:
    image_path: Path | None = None
    pixel_size: float | None = None  # calibrated units (e.g. microns) per pixel
    measurements: pd.DataFrame | None = None
    last_run: LastRun | None = None
    engine: SegmentationEngine = field(default_factory=SegmentationEngine)

    def export_base_path(self) -> Path:
        """Default base path (no extension) for exported files."""
        if self.image_path is not None:
            return self.image_path.with_suffix("")
        return Path.cwd() / "cose_cell_segmenter_export"
