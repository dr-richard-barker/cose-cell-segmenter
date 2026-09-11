"""Standard image-processing panel.

Applies an operation to a chosen Image layer and adds the result as a new,
non-destructive "<name>_processed" layer — the original is never modified.
Cropping to a region of interest happens later, at segmentation time (see
cellpose_panel.py), so preprocessing can be tried once and reused across
several ROI segmentation runs.
"""

from __future__ import annotations

import napari
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cose_cell_segmenter.core import preprocessing as pp

#: name -> (function, default param value, param label)
OPERATIONS = {
    "Gaussian blur": (pp.gaussian_blur, "sigma", 1.0, "Sigma (px)"),
    "Median blur": (pp.median_blur, "radius", 2, "Radius (px)"),
    "CLAHE (local contrast)": (pp.clahe, "clip_limit", 0.01, "Clip limit"),
    "Rolling-ball background subtract": (
        pp.rolling_ball_bg_subtract,
        "radius",
        50,
        "Ball radius (px)",
    ),
}


class PreprocessPanel(QWidget):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.source_combo = QComboBox()
        form.addRow("Source layer", self.source_combo)

        self.op_combo = QComboBox()
        self.op_combo.addItems(list(OPERATIONS))
        self.op_combo.currentTextChanged.connect(self._on_op_changed)
        form.addRow("Operation", self.op_combo)

        self.param_spin = QDoubleSpinBox()
        self.param_spin.setDecimals(3)
        self.param_spin.setRange(0.0, 10000.0)
        self.param_label = QLabel()
        form.addRow(self.param_label, self.param_spin)

        layout.addLayout(form)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.apply_btn)
        layout.addStretch()

        self._on_op_changed(self.op_combo.currentText())
        self._refresh_sources()
        for event in (viewer.layers.events.inserted, viewer.layers.events.removed, viewer.layers.events.renamed):
            event.connect(lambda *_a, **_k: self._refresh_sources())

    def _refresh_sources(self) -> None:
        current = self.source_combo.currentText()
        self.source_combo.clear()
        names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        self.source_combo.addItems(names)
        if current in names:
            self.source_combo.setCurrentText(current)

    def _on_op_changed(self, op_name: str) -> None:
        _func, _param_name, default, label = OPERATIONS[op_name]
        self.param_label.setText(label)
        self.param_spin.setValue(default)

    def _on_apply_clicked(self) -> None:
        source_name = self.source_combo.currentText()
        if not source_name or source_name not in self.viewer.layers:
            return
        layer = self.viewer.layers[source_name]

        op_name = self.op_combo.currentText()
        func, param_name, _default, _label = OPERATIONS[op_name]
        kwargs = {param_name: self.param_spin.value()}
        # napari doesn't retain a `channel_axis` attribute on the Image layer
        # after add_image(); treat plain >2D non-RGB data as channel-last by
        # convention (matches this app's own convention throughout).
        is_rgb = getattr(layer, "rgb", False)
        channel_axis = -1 if (layer.data.ndim > 2 and not is_rgb) else None

        result = func(layer.data, channel_axis=channel_axis, **kwargs)

        out_name = f"{source_name}_processed"
        if out_name in self.viewer.layers:
            self.viewer.layers[out_name].data = result
        else:
            self.viewer.add_image(result, name=out_name)
