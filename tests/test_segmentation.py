import numpy as np

from cose_cell_segmenter.core.segmentation import BUILTIN_MODELS, paste_into_canvas


def test_builtin_models_nonempty():
    # sanity check that we're reading real model names from the installed
    # cellpose version, not an empty/stale list
    assert len(BUILTIN_MODELS) > 0
    assert all(isinstance(name, str) for name in BUILTIN_MODELS)


def test_paste_into_canvas_first_paste_keeps_labels():
    canvas = np.zeros((10, 10), dtype=np.int32)
    roi_masks = np.array([[0, 1], [1, 2]], dtype=np.int32)
    paste_into_canvas(canvas, roi_masks, bbox=(2, 2, 4, 4))
    assert np.array_equal(canvas[2:4, 2:4], roi_masks)
    assert canvas.max() == 2


def test_paste_into_canvas_second_paste_relabels_to_avoid_collision():
    canvas = np.zeros((10, 10), dtype=np.int32)
    roi_a = np.array([[1, 1], [1, 2]], dtype=np.int32)
    roi_b = np.array([[1, 2], [0, 0]], dtype=np.int32)

    paste_into_canvas(canvas, roi_a, bbox=(0, 0, 2, 2))
    assert canvas.max() == 2

    paste_into_canvas(canvas, roi_b, bbox=(5, 5, 7, 7))
    # roi_b's labels 1,2 must be relabeled to continue after canvas's max (2)
    # -> 3,4, so the two regions never share a label id
    assert canvas.max() == 4
    assert set(np.unique(canvas[5:7, 5:7])) - {0} == {3, 4}
    # first region untouched
    assert np.array_equal(canvas[0:2, 0:2], roi_a)
