"""Per-object measurements panel — regionprops_table against the "masks" layer."""

from __future__ import annotations

import napari
import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cose_cell_segmenter.core import measurements as meas
from cose_cell_segmenter.widgets.cellpose_panel import MASKS_LAYER_NAME
from cose_cell_segmenter.widgets.state import AppState


class MeasurePanel(QWidget):
    def __init__(self, viewer: napari.Viewer, state: AppState, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.state = state

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.intensity_combo = QComboBox()
        self.intensity_combo.addItem("(none)")
        form.addRow("Intensity source", self.intensity_combo)
        layout.addLayout(form)

        self.measure_btn = QPushButton("Measure")
        self.measure_btn.clicked.connect(self._on_measure_clicked)
        layout.addWidget(self.measure_btn)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        self._refresh_sources()
        for event in (viewer.layers.events.inserted, viewer.layers.events.removed, viewer.layers.events.renamed):
            event.connect(lambda *_a, **_k: self._refresh_sources())

    def _refresh_sources(self) -> None:
        current = self.intensity_combo.currentText()
        self.intensity_combo.clear()
        self.intensity_combo.addItem("(none)")
        names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        self.intensity_combo.addItems(names)
        if current in names or current == "(none)":
            self.intensity_combo.setCurrentText(current)

    def _on_measure_clicked(self) -> None:
        if MASKS_LAYER_NAME not in self.viewer.layers:
            return
        masks = np.asarray(self.viewer.layers[MASKS_LAYER_NAME].data)

        intensity = None
        intensity_name = self.intensity_combo.currentText()
        if intensity_name and intensity_name != "(none)" and intensity_name in self.viewer.layers:
            intensity = np.asarray(self.viewer.layers[intensity_name].data)

        df = meas.measure(masks, intensity_image=intensity, pixel_size=self.state.pixel_size)
        self.state.measurements = df
        self._populate_table(df)

    def _populate_table(self, df) -> None:
        self.table.clear()
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(list(df.columns))
        self.table.setRowCount(len(df))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, value in enumerate(row):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
