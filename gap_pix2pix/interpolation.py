"""Linear interpolation across detector gaps.

The generator does not fill the gaps from scratch: it refines a linear interpolation. Detector
gaps are complete rows and columns of the image ("grid-like" masks), so the interpolation is
separable: gap columns are interpolated along the rows, then gap rows along the columns. Where
a gap row crosses a gap column this gives bilinear interpolation from the four surrounding
valid pixels. Gaps at the image border take the value of the nearest valid row/column.
"""
import numpy as np


def _neighbours(gap_1d):
    """Gap indices, their nearest valid neighbours on both sides and the linear weights."""
    valid = np.flatnonzero(~gap_1d)
    if len(valid) == 0:
        raise ValueError("every row/column of the image is a gap")
    idx = np.flatnonzero(gap_1d)
    right = np.clip(np.searchsorted(valid, idx), 0, len(valid) - 1)
    left = np.clip(np.searchsorted(valid, idx) - 1, 0, len(valid) - 1)
    lo, hi = valid[left], valid[right]
    w = np.where(hi > lo, (idx - lo) / np.maximum(hi - lo, 1), 0.0).astype(np.float32)
    return idx, lo, hi, w


def split_grid_mask(gap_mask):
    """(gap rows, gap columns) of a grid-like gap mask (True = gap); raises otherwise."""
    gap_mask = np.asarray(gap_mask, dtype=bool)
    rows, cols = gap_mask.all(axis=1), gap_mask.all(axis=0)
    if not np.array_equal(gap_mask, rows[:, None] | cols[None, :]):
        raise ValueError("the gap mask must consist of complete rows and columns")
    return rows, cols


def interpolate_gaps(images, gap_mask):
    """Linearly interpolate the gaps of an image (H, W) or a stack of images (..., H, W).

    Args:
        images: image(s); values in the gaps are ignored.
        gap_mask: (H, W) bool, True = gap, made of complete rows and columns.

    Returns:
        float32 copy of `images` with the gaps filled.
    """
    rows, cols = split_grid_mask(gap_mask)
    out = np.array(images, dtype=np.float32, copy=True)
    if cols.any():
        idx, lo, hi, w = _neighbours(cols)
        out[..., idx] = (1 - w) * out[..., lo] + w * out[..., hi]
    if rows.any():
        idx, lo, hi, w = _neighbours(rows)
        out[..., idx, :] = (1 - w)[:, None] * out[..., lo, :] + w[:, None] * out[..., hi, :]
    return out
