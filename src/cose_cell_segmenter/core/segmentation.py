"""Thin wrapper around cellpose.models.CellposeModel.

No Qt/napari imports here. `SegmentationEngine` caches the loaded model
instance so switching parameters and re-running doesn't reload the (large)
pretrained weights every time — only a change of model name or GPU flag
triggers a reload.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cellpose import models

#: Built-in pretrained model names, as reported by the installed cellpose version.
BUILTIN_MODELS: tuple[str, ...] = tuple(models.MODEL_NAMES)


@dataclass
class SegmentParams:
    """Cellpose `.eval()` parameters exposed in the UI."""

    model_name: str = "cpsam_v2"  # one of BUILTIN_MODELS, or a path to custom weights
    gpu: bool = True
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    channel_axis: int | None = None
    do_3D: bool = False
    anisotropy: float | None = None


class SegmentationEngine:
    """Loads and caches a CellposeModel; runs it on a given image array."""

    def __init__(self) -> None:
        self._model: models.CellposeModel | None = None
        self._model_key: tuple[str, bool] | None = None

    def _get_model(self, model_name: str, gpu: bool) -> models.CellposeModel:
        key = (model_name, gpu)
        if self._model is None or self._model_key != key:
            self._model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
            self._model_key = key
        return self._model

    def run(
        self, image: np.ndarray, params: SegmentParams, progress=None
    ) -> tuple[np.ndarray, list]:
        """Segment `image` (already cropped to the region of interest, if any).

        Returns `(masks, flows)` — see `cellpose.models.CellposeModel.eval` for
        their exact shapes. `masks` is a labeled integer array, 0 = background.
        """
        model = self._get_model(params.model_name, params.gpu)
        masks, flows, _styles = model.eval(
            image,
            channel_axis=params.channel_axis,
            diameter=params.diameter,
            flow_threshold=params.flow_threshold,
            cellprob_threshold=params.cellprob_threshold,
            min_size=params.min_size,
            do_3D=params.do_3D,
            anisotropy=params.anisotropy,
            progress=progress,
        )
        return masks, flows


def paste_into_canvas(
    canvas: np.ndarray, roi_masks: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    """Paste a cropped ROI's label mask into a full-size canvas at `bbox`.

    ROI labels are relabeled to continue after whatever's already in `canvas`,
    so results from multiple segmented regions accumulate without ID
    collisions. `bbox` is `(row_start, col_start, row_end, col_end)` in
    canvas coordinates. Modifies and returns `canvas`.
    """
    row0, col0, row1, col1 = bbox
    next_id = int(canvas.max()) + 1
    region = canvas[row0:row1, col0:col1]
    roi_nonzero = roi_masks > 0
    region[roi_nonzero] = roi_masks[roi_nonzero] + next_id - 1
    return canvas
