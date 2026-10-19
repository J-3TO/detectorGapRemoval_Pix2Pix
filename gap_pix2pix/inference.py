"""Inpainting detector gaps with a trained generator.

`inpaint` fills the gaps of every image of a stack (projections, or any other 2D images):

    1. z-score the image with the mean / std of its valid pixels
    2. interpolate the gaps linearly
    3. run the generator on the whole image (it corrects the interpolation inside the gaps)
    4. undo the z-score; valid pixels are returned unchanged

`inpaint_sinograms` does the same in the sinogram domain, see there.
"""
import numpy as np
import torch

from .interpolation import interpolate_gaps, split_grid_mask


@torch.no_grad()
def inpaint(images, gap_mask, generator, batch_size=1, progress=False):
    """Fill the gaps of an image (H, W) or a stack of images (N, H, W).

    Args:
        images: the image(s); values in the gaps are ignored.
        gap_mask: (H, W) bool, True = gap, made of complete rows and columns (as detector gaps).
        generator: a `UNetMask`, e.g. from `weights.load_generator`.
        batch_size: images per forward pass.
        progress: print progress.

    Returns:
        float32 array of the shape of `images`, gaps filled.
    """
    gap_mask = np.asarray(gap_mask, dtype=bool)
    images = np.asarray(images, dtype=np.float32)
    single = images.ndim == 2
    stack = images[None] if single else images
    if stack.shape[-2:] != gap_mask.shape:
        raise ValueError(f"image shape {stack.shape[-2:]} != mask shape {gap_mask.shape}")
    if min(gap_mask.shape) < 16:
        raise ValueError(f"images must be at least 16 x 16 pixels (U-Net depth), got {gap_mask.shape}")
    if not gap_mask.any():
        return images.copy()
    split_grid_mask(gap_mask)  # raises unless the gaps are complete rows / columns

    device = next(generator.parameters()).device
    valid = torch.from_numpy((~gap_mask).astype(np.float32)).to(device)[None, None]
    out = np.empty_like(stack)
    for start in range(0, len(stack), batch_size):
        batch = stack[start:start + batch_size]
        mean = batch[:, ~gap_mask].mean(axis=1)[:, None, None]
        std = np.maximum(batch[:, ~gap_mask].std(axis=1), 1e-12)[:, None, None]
        x = interpolate_gaps((batch - mean) / std, gap_mask)
        y = generator(torch.from_numpy(x[:, None]).to(device), valid.expand(len(x), -1, -1, -1))
        y = y[:, 0].cpu().numpy() * std + mean
        out[start:start + batch_size] = np.where(gap_mask, y, batch)
        if progress:
            print(f"\r{min(start + batch_size, len(stack))}/{len(stack)}", end="", flush=True)
    if progress:
        print()
    return out[0] if single else out


def inpaint_sinograms(projections, gap_mask, generator, batch_size=1, rows_per_chunk=64,
                      progress=False):
    """Fill the gaps of a projection stack (angles, H, W) in two passes.

    1. Gap columns in the sinogram domain: for every detector row that is not a gap row, its
       sinogram (angles, W) is inpainted; there a gap column is a stripe across all angles, with
       the object's trajectories continuing on both sides.
    2. Gap rows in the projection domain: a gap row is missing at every angle (a whole sinogram is
       missing), so these rows, including their crossings with the gap columns, are inpainted per
       projection, with the gap columns already filled by pass 1.

    The stack is processed `rows_per_chunk` detector rows at a time, to bound the memory of the
    transposed copies. Returns a float32 copy of `projections` with the gaps filled.
    """
    out = np.array(projections, dtype=np.float32, copy=True)
    gap_rows, gap_cols = split_grid_mask(gap_mask)
    if gap_cols.any():
        sino_mask = np.broadcast_to(gap_cols, (out.shape[0], out.shape[2]))
        rows = np.flatnonzero(~gap_rows)
        for i in range(0, len(rows), rows_per_chunk):
            r = rows[i:i + rows_per_chunk]
            sinograms = np.moveaxis(out[:, r], 1, 0)  # (rows, angles, W)
            out[:, r] = np.moveaxis(inpaint(sinograms, sino_mask, generator, batch_size), 0, 1)
            if progress:
                print(f"\rsinograms: rows {r[-1] + 1}/{out.shape[1]}", end="", flush=True)
        if progress:
            print()
    if gap_rows.any():
        row_mask = np.broadcast_to(gap_rows[:, None], gap_mask.shape)
        out = inpaint(out, row_mask, generator, batch_size, progress=progress)
    return out
