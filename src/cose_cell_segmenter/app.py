"""CoSE Cell Segmenter — entry point.

Builds a napari Viewer and docks the five panels (load, preprocess, cellpose,
measure, export) around it. Run with `uv run cose-cell-segmenter`, or
`python -m cose_cell_segmenter.app`.
"""

from __future__ import annotations

from dataclasses import dataclass

import napari

from cose_cell_segmenter.widgets.cellpose_panel import CellposePanel
from cose_cell_segmenter.widgets.export_panel import ExportPanel
from cose_cell_segmenter.widgets.load_panel import LoadPanel
from cose_cell_segmenter.widgets.measure_panel import MeasurePanel
from cose_cell_segmenter.widgets.preprocess_panel import PreprocessPanel
from cose_cell_segmenter.widgets.state import AppState


@dataclass
class Panels:
    load: LoadPanel
    preprocess: PreprocessPanel
    cellpose: CellposePanel
    measure: MeasurePanel
    export: ExportPanel


def build_viewer(show: bool = True) -> tuple[napari.Viewer, AppState, Panels]:
    """Construct the viewer + all panels. Returns (viewer, state, panels) so a
    caller (or a test) can drive the app programmatically without going
    through the Qt event loop or digging through napari's dock-widget
    internals.
    """
    viewer = napari.Viewer(title="CoSE Cell Segmenter", show=show)
    state = AppState()

    load_panel = LoadPanel(state)
    load_panel.image_loaded.connect(lambda path: viewer.add_image(load_panel.last_array, name=path.stem))

    panels = Panels(
        load=load_panel,
        preprocess=PreprocessPanel(viewer),
        cellpose=CellposePanel(viewer, state),
        measure=MeasurePanel(viewer, state),
        export=ExportPanel(viewer, state),
    )

    viewer.window.add_dock_widget(panels.load, name="Load image", area="right")
    viewer.window.add_dock_widget(panels.preprocess, name="Preprocess", area="right")
    viewer.window.add_dock_widget(panels.cellpose, name="Cellpose", area="right")
    viewer.window.add_dock_widget(panels.measure, name="Measure", area="right")
    viewer.window.add_dock_widget(panels.export, name="Export", area="right")

    return viewer, state, panels


def main() -> None:
    viewer, _state, _panels = build_viewer(show=True)
    napari.run()


if __name__ == "__main__":
    main()
