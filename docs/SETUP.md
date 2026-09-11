# Setup

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — manages an isolated Python 3.12 for this project; doesn't
  touch your system Python.

## Install and run

```bash
cd cose-cell-segmenter
uv sync              # creates .venv/, installs cellpose, napari[pyqt6], scikit-image, pandas, tifffile
uv run cose-cell-segmenter
```

`uv run pytest` runs the unit tests (`tests/`) — these cover the pure-function core
(`src/cose_cell_segmenter/core/`: preprocessing, measurements, mask-canvas stitching) and need
no GPU and no model download.

## First run / model download

Cellpose downloads the chosen model's pretrained weights to `~/.cellpose/models/` the first time
you segment with it, not at install time. The default model, `cpsam_v2`, is about 1.1 GB —
expect a pause the first time only; subsequent runs reuse the cached weights.

## Apple Silicon (MPS)

On Apple Silicon Macs, Cellpose auto-detects and uses the GPU (MPS) when the "Use GPU" checkbox
in the Cellpose panel is ticked (it is by default) — no CUDA or extra setup needed, as long as
`uv sync` installed a normal `pip install torch` (which ships MPS support on macOS already). Mask
post-processing (not the neural-net forward pass) runs on CPU on Mac — a minor performance
nuance, not a functional limitation.

## Project layout

See the top-level `README.md` for the workflow; briefly:

- `src/cose_cell_segmenter/core/` — pure functions (no Qt/napari imports), independently
  testable: `segmentation.py` (Cellpose wrapper + model caching), `preprocessing.py`,
  `measurements.py` (regionprops), `io_export.py` (mask/ROI/CSV/project export).
- `src/cose_cell_segmenter/widgets/` — the napari dock panels (thin — they call into `core/`).
- `src/cose_cell_segmenter/app.py` — builds the napari `Viewer` and docks the panels.

## License note

Cellpose's code is BSD-3-Clause. Its **pretrained model weights** were trained on CC-BY-NC
(non-commercial) data per Cellpose's own documentation — fine for personal/research use, worth
checking before any commercial use.
