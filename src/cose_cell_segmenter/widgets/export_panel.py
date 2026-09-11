"""Export panel — masks, ImageJ ROIs, measurements CSV, and a reopenable project file."""

from __future__ import annotations

from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cose_cell_segmenter.core import io_export
from cose_cell_segmenter.widgets.cellpose_panel import MASKS_LAYER_NAME
from cose_cell_segmenter.widgets.state import AppState


class ExportPanel(QWidget):
    def __init__(self, viewer: napari.Viewer, state: AppState, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.state = state

        layout = QVBoxLayout(self)

        self.masks_btn = QPushButton("Save masks (PNG, all segmented regions)")
        self.masks_btn.clicked.connect(self._on_save_masks)
        layout.addWidget(self.masks_btn)

        self.rois_btn = QPushButton("Save ImageJ ROIs (.zip, all segmented regions)")
        self.rois_btn.clicked.connect(self._on_save_rois)
        layout.addWidget(self.rois_btn)

        self.csv_btn = QPushButton("Save measurements (CSV)")
        self.csv_btn.clicked.connect(self._on_save_csv)
        layout.addWidget(self.csv_btn)

        self.project_btn = QPushButton("Save project (_seg.npy, last run only)")
        self.project_btn.clicked.connect(self._on_save_project)
        layout.addWidget(self.project_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()

    # -- helpers -----------------------------------------------------------

    def _masks(self) -> np.ndarray | None:
        if MASKS_LAYER_NAME not in self.viewer.layers:
            self._warn("Nothing to export yet — run Segment first.")
            return None
        return np.asarray(self.viewer.layers[MASKS_LAYER_NAME].data)

    def _choose_base_path(self) -> Path | None:
        default = str(self.state.export_base_path())
        path_str, _ = QFileDialog.getSaveFileName(self, "Export base name", default)
        return Path(path_str) if path_str else None

    def _warn(self, message: str) -> None:
        self.status_label.setText(message)

    def _info(self, message: str) -> None:
        self.status_label.setText(message)

    # -- actions -------------------------------------------------------

    def _on_save_masks(self) -> None:
        masks = self._masks()
        if masks is None:
            return
        base = self._choose_base_path()
        if base is None:
            return
        written = io_export.save_masks_array(masks, base, png=True)
        self._info(f"Saved masks to {', '.join(str(p) for p in written)}")

    def _on_save_rois(self) -> None:
        masks = self._masks()
        if masks is None:
            return
        base = self._choose_base_path()
        if base is None:
            return
        out_path = io_export.save_rois(masks, base)
        if out_path is None:
            self._warn("No masks to export as ROIs.")
        else:
            self._info(f"Saved ImageJ ROIs to {out_path}")

    def _on_save_csv(self) -> None:
        if self.state.measurements is None:
            self._warn("No measurements yet — run Measure first.")
            return
        base = self._choose_base_path()
        if base is None:
            return
        out_path = io_export.save_measurements_csv(self.state.measurements, f"{base}_measurements.csv")
        self._info(f"Saved measurements to {out_path}")

    def _on_save_project(self) -> None:
        run = self.state.last_run
        if run is None:
            self._warn("No segmentation run yet — run Segment first.")
            return
        base = self._choose_base_path()
        if base is None:
            return
        out_path = io_export.save_project(run.image, run.masks, run.flows, base)
        self._info(f"Saved project (last segmented region only) to {out_path}")
