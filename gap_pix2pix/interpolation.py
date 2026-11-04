"""Linear interpolation across detector gaps.

The generator does not fill the gaps from scratch: it refines a linear interpolation. Detector
gaps are complete rows and columns of the image ("grid-like" masks), so the interpolation is
separable: gap columns are interpolated along the rows, then gap rows along the columns. Where
a gap row crosses a gap column this gives bilinear interpolation from the four surrounding
valid pixels. Gaps at the image border take the value of the nearest valid row/column.

`interpolate_gaps` (NumPy) prepares the training patches; `interpolate_gaps_torch` computes the
same on a batch of tensors, for inference (`UNetMask.predict`).
"""
import numpy as np
import torch


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


def _interpolate_1d_torch(x, gap, dim):
    """Linear interpolation of x along `dim` at the positions where `gap` (bool, x's shape with
    every axis but `dim` of size 1 or full) is True, from the nearest valid neighbours."""
    n = x.shape[dim]
    shape = [1] * x.ndim
    shape[dim] = n
    pos = torch.arange(n, device=x.device).view(shape).expand_as(gap)
    left = torch.where(gap, -1, pos).cummax(dim).values
    right = torch.where(gap, n, pos).flip(dim).cummin(dim).values.flip(dim)
    left = torch.where(left < 0, right, left)       # gap at the start: nearest valid on the right
    right = torch.where(right >= n, left, right)    # gap at the end: nearest valid on the left
    w = torch.where(right > left, (pos - left) / (right - left).clamp(min=1), 0.0).to(x.dtype)
    left, right, w = (t.expand_as(x) for t in (left, right, w))
    filled = (1 - w) * x.gather(dim, left) + w * x.gather(dim, right)
    return torch.where(gap.expand_as(x), filled, x)


def interpolate_gaps_torch(x, valid):
    """`interpolate_gaps` for a batch of tensors.

    Args:
        x: (B, C, H, W) images; values in the gaps are ignored.
        valid: (B, 1, H, W) or (1, 1, H, W), 1 = valid pixel, 0 = gap; the gaps of every image
            must be complete rows and columns (they may differ between images).

    Returns:
        x with the gaps filled (gap columns along the rows, then gap rows along the columns).
    """
    gap = valid == 0
    cols, rows = gap.all(dim=-2, keepdim=True), gap.all(dim=-1, keepdim=True)  # (B,1,1,W), (B,1,H,1)
    if not torch.equal(gap, cols | rows):
        raise ValueError("the gap mask must consist of complete rows and columns")
    if (~cols).sum(dim=-1).min() == 0 or (~rows).sum(dim=-2).min() == 0:
        raise ValueError("every row/column of the image is a gap")
    x = torch.where(gap, 0.0, x)  # NaN or other values in the gaps must not leak into the result
    x = _interpolate_1d_torch(x, cols, dim=-1)
    return _interpolate_1d_torch(x, rows, dim=-2)
