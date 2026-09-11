"""Cellpose model/parameter panel — the "Segment" button.

Reads the last-drawn rectangle on the "ROI" Shapes layer (creating it if
missing) to crop before running; if nothing's been drawn, segments the full
image. Results are painted into a "masks" Labels layer (created/resized as
needed) at the correct offset, via core.segmentation.paste_into_canvas.
"""

from __future__ import annotations

import numpy as np
import napari
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cose_cell_segmenter.core.segmentation import BUILTIN_MODELS, SegmentParams, paste_into_canvas
from cose_cell_segmenter.widgets.state import AppState, LastRun

CUSTOM_MODEL_LABEL = "Custom weights…"
ROI_LAYER_NAME = "ROI"
MASKS_LAYER_NAME = "masks"


class CellposePanel(QWidget):
    def __init__(self, viewer: napari.Viewer, state: AppState, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.state = state
        self._custom_model_path: str | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.source_combo = QComboBox()
        form.addRow("Source layer", self.source_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems([*BUILTIN_MODELS, CUSTOM_MODEL_LABEL])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        form.addRow("Model", self.model_combo)

        self.gpu_check = QCheckBox("Use GPU (CUDA/MPS)")
        self.gpu_check.setChecked(True)
        form.addRow(self.gpu_check)

        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0.0, 2000.0)
        self.diameter_spin.setSpecialValueText("auto")
        self.diameter_spin.setValue(0.0)
        form.addRow("Diameter (px)", self.diameter_spin)

        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setDecimals(2)
        self.flow_spin.setRange(0.0, 3.0)
        self.flow_spin.setSingleStep(0.05)
        self.flow_spin.setValue(0.4)
        form.addRow("Flow threshold", self.flow_spin)

        self.cellprob_spin = QDoubleSpinBox()
        self.cellprob_spin.setDecimals(2)
        self.cellprob_spin.setRange(-6.0, 6.0)
        self.cellprob_spin.setSingleStep(0.1)
        self.cellprob_spin.setValue(0.0)
        form.addRow("Cell probability threshold", self.cellprob_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 100000)
        self.min_size_spin.setValue(15)
        form.addRow("Min size (px)", self.min_size_spin)

        layout.addLayout(form)

        self.roi_label = QLabel("No region selected — will segment the full image")
        self.roi_label.setWordWrap(True)
        layout.addWidget(self.roi_label)

        self.segment_btn = QPushButton("Segment")
        self.segment_btn.clicked.connect(self._on_segment_clicked)
        layout.addWidget(self.segment_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        self._ensure_roi_layer()
        self._refresh_sources()
        for event in (viewer.layers.events.inserted, viewer.layers.events.removed, viewer.layers.events.renamed):
            event.connect(lambda *_a, **_k: self._refresh_sources())

    # -- layer bookkeeping -------------------------------------------------

    def _ensure_roi_layer(self):
        if ROI_LAYER_NAME not in self.viewer.layers:
            self.viewer.add_shapes(name=ROI_LAYER_NAME, shape_type="rectangle", edge_color="yellow", face_color="transparent")
        return self.viewer.layers[ROI_LAYER_NAME]

    def _refresh_sources(self) -> None:
        current = self.source_combo.currentText()
        self.source_combo.clear()
        names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        self.source_combo.addItems(names)
        if current in names:
            self.source_combo.setCurrentText(current)

    def _on_model_changed(self, text: str) -> None:
        if text == CUSTOM_MODEL_LABEL:
            path, _ = QFileDialog.getOpenFileName(self, "Select custom Cellpose weights")
            if path:
                self._custom_model_path = path
            else:
                # user cancelled — fall back to the default built-in model
                self.model_combo.setCurrentText(BUILTIN_MODELS[0])

    def _current_model_name(self) -> str:
        text = self.model_combo.currentText()
        if text == CUSTOM_MODEL_LABEL and self._custom_model_path:
            return self._custom_model_path
        return text

    # -- ROI bounding box ----------------------------------------------

    def _current_roi_bbox(self, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
        """Bounding box of the last-drawn shape on the ROI layer, clipped to
        `image_shape`, as (row0, col0, row1, col1) — or None if nothing's drawn.
        """
        roi_layer = self.viewer.layers[ROI_LAYER_NAME] if ROI_LAYER_NAME in self.viewer.layers else None
        if roi_layer is None or len(roi_layer.data) == 0:
            return None
        vertices = roi_layer.data[-1]  # most recently drawn shape
        rows = vertices[:, 0]
        cols = vertices[:, 1]
        row0 = int(np.clip(np.floor(rows.min()), 0, image_shape[0]))
        row1 = int(np.clip(np.ceil(rows.max()), 0, image_shape[0]))
        col0 = int(np.clip(np.floor(cols.min()), 0, image_shape[1]))
        col1 = int(np.clip(np.ceil(cols.max()), 0, image_shape[1]))
        if row1 <= row0 or col1 <= col0:
            return None
        return row0, col0, row1, col1

    # -- segmentation ----------------------------------------------------

    def _on_segment_clicked(self) -> None:
        source_name = self.source_combo.currentText()
        if not source_name or source_name not in self.viewer.layers:
            return
        source_layer = self.viewer.layers[source_name]
        image = np.asarray(source_layer.data)
        spatial_shape = image.shape[:2]

        bbox = self._current_roi_bbox(spatial_shape)
        if bbox is None:
            bbox = (0, 0, spatial_shape[0], spatial_shape[1])
            self.roi_label.setText("No region selected — segmenting the full image")
        else:
            self.roi_label.setText(f"Segmenting region rows {bbox[0]}:{bbox[2]}, cols {bbox[1]}:{bbox[3]}")

        row0, col0, row1, col1 = bbox
        crop = image[row0:row1, col0:col1]

        is_rgb = getattr(source_layer, "rgb", False)
        channel_axis = -1 if (image.ndim > 2 and not is_rgb) else None

        params = SegmentParams(
            model_name=self._current_model_name(),
            gpu=self.gpu_check.isChecked(),
            diameter=self.diameter_spin.value() or None,
            flow_threshold=self.flow_spin.value(),
            cellprob_threshold=self.cellprob_spin.value(),
            min_size=self.min_size_spin.value(),
            channel_axis=channel_axis,
        )

        self.progress_bar.setValue(0)
        self.segment_btn.setEnabled(False)
        try:
            masks, flows = self.state.engine.run(crop, params, progress=self.progress_bar)
        finally:
            self.segment_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        self.state.last_run = LastRun(image=crop, masks=masks, flows=flows)
        self._paint_masks(masks, bbox, spatial_shape)

    def _paint_masks(self, roi_masks: np.ndarray, bbox: tuple[int, int, int, int], spatial_shape: tuple[int, int]) -> None:
        if MASKS_LAYER_NAME in self.viewer.layers:
            masks_layer = self.viewer.layers[MASKS_LAYER_NAME]
            if masks_layer.data.shape != spatial_shape:
                masks_layer.data = np.zeros(spatial_shape, dtype=np.int32)
        else:
            masks_layer = self.viewer.add_labels(np.zeros(spatial_shape, dtype=np.int32), name=MASKS_LAYER_NAME)

        canvas = np.asarray(masks_layer.data).copy()
        paste_into_canvas(canvas, roi_masks, bbox)
        masks_layer.data = canvas
