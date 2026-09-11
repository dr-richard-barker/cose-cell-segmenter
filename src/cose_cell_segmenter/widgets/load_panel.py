"""Image loading + pixel-size calibration panel."""

from __future__ import annotations

from pathlib import Path

from cellpose import io as cp_io
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cose_cell_segmenter.widgets.state import AppState

IMAGE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"


class LoadPanel(QWidget):
    """Emits `image_loaded(path)` after successfully reading a file into `state`."""

    image_loaded = Signal(Path)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._last_array = None

        layout = QVBoxLayout(self)

        self.open_btn = QPushButton("Open image…")
        self.open_btn.clicked.connect(self._on_open_clicked)
        layout.addWidget(self.open_btn)

        self.path_label = QLabel("No image loaded")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        form = QFormLayout()
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setDecimals(4)
        self.pixel_size_spin.setRange(0.0, 10000.0)
        self.pixel_size_spin.setSpecialValueText("uncalibrated (px)")
        self.pixel_size_spin.setValue(0.0)
        self.pixel_size_spin.valueChanged.connect(self._on_pixel_size_changed)
        form.addRow("Pixel size (units/px)", self.pixel_size_spin)
        layout.addLayout(form)

        layout.addStretch()

    def _on_pixel_size_changed(self, value: float) -> None:
        self.state.pixel_size = value if value > 0 else None

    def _on_open_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open image", "", IMAGE_FILTER)
        if not path_str:
            return
        self.load_path(Path(path_str))

    def load_path(self, path: Path):
        """Read `path` into a numpy array, update state, and return it."""
        array = cp_io.imread(str(path))
        self.state.image_path = path
        self._last_array = array
        self.path_label.setText(str(path))
        self.image_loaded.emit(path)
        return array

    @property
    def last_array(self):
        return self._last_array
