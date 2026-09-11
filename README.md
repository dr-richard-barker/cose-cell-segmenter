# CoSE Cell Segmenter

A desktop GUI for running [Cellpose](https://github.com/mouseland/cellpose) (deep-learning
cell/nucleus/spore segmentation) — draw a region of interest, optionally preprocess it, tune
Cellpose's own parameters, segment, hand-correct the result, measure, and export. Part of the
CoSE astrobotany tool family, but a local desktop app rather than a web-deployed one — Cellpose
needs GPU inference and a large PyTorch stack, which doesn't fit a static site.

## What it does

Built on [napari](https://napari.org) rather than a from-scratch viewer, so you get napari's own
pan/zoom image viewer, region drawing (`Shapes` layer), and mask paint/erase/merge/split editing
(`Labels` layer) for free, instead of reimplementing them.

1. **Load** an image (any format `cellpose.io.imread`/tifffile/imageio can read: TIFF, PNG, JPEG,
   multi-page TIFF, …) and optionally set a pixel-size calibration (units per pixel).
2. **Draw a region** on the yellow "ROI" shapes layer (rectangle tool) — or skip it to segment
   the whole image.
3. **Preprocess** (optional): Gaussian blur, median blur, CLAHE local-contrast, or rolling-ball
   background subtraction, applied to a chosen image layer and written to a new
   `<name>_processed` layer. Non-destructive — the original is never touched, and you pick which
   layer (raw or processed) actually gets segmented.
4. **Segment**: pick a Cellpose model (`cpsam_v2`, `cpsam`, `cpdino`, `cpdino-vitb`, or your own
   trained weights via "Custom weights…"), set diameter/flow threshold/cell-probability
   threshold/min size, and run. GPU (CUDA or Apple Silicon MPS) is used automatically when
   available. The model stays loaded between runs, so changing a parameter and re-running is
   fast. Results from multiple regions accumulate into one "masks" layer without ID collisions.
5. **Correct by hand**: napari's own `Labels` layer tools — paint, erase, fill, merge — work
   directly on the "masks" layer, the same capability Cellpose's own GUI offers.
6. **Measure**: per-object table (area, centroid, perimeter, eccentricity, solidity, equivalent
   diameter, and intensity stats if you pick an intensity-source layer) via
   `skimage.measure.regionprops_table` — Cellpose itself only produces masks, not measurements.
7. **Export**: labeled mask as PNG/TIFF, an ImageJ/Fiji-importable ROI `.zip`, a measurements
   CSV, and a `_seg.npy` project file (Cellpose's own native format — reopenable in this app or
   in Cellpose's official GUI; scoped to the single most recent segmentation run, since it needs
   real flow data that only exists per-run, not for the accumulated multi-region canvas).

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync
uv run cose-cell-segmenter
```

The first segmentation run downloads the chosen model's pretrained weights (~1.1 GB for
`cpsam_v2`) to `~/.cellpose/models/` — expect a pause the first time only.

See [`docs/SETUP.md`](docs/SETUP.md) for more (Apple Silicon/MPS notes, running the tests).

## License

Code is BSD-3-Clause (see [`LICENSE`](LICENSE)), matching Cellpose's own license.

**Cellpose's pretrained model weights are licensed separately and more restrictively**: per
Cellpose's own documentation, they're trained on CC-BY-NC (non-commercial) data. That's a
non-issue for personal/research use — flagging it because it's a real constraint if this were
ever used in a commercial context.

## Status

Core pipeline (segmentation, preprocessing, measurements, export) is covered by unit tests
(`uv run pytest`) and was verified end-to-end with a real Cellpose run on Apple Silicon MPS. The
Qt/napari widget layer has not been visually driven/screenshotted by an automated tool — it was
exercised programmatically (calling the same handlers the UI buttons call) rather than
interactively, so give it a manual smoke test before relying on it for real work.
