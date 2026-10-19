"""Gap masks: the real gaps of a tiled detector, and the synthetic gaps used for training.

Convention throughout this package: a gap mask is a bool array with True = gap. The generator
itself gets the inverse as its second input channel (1 = valid pixel, 0 = gap).
"""
import numpy as np

# DECTRIS Pilatus3 2M: 3 x 8 modules of 487 x 195 pixels, separated by 7 / 17 pixel gaps
PILATUS3_2M = dict(module_size=(487, 195), gap_size=(7, 17), grid=(3, 8))


def detector_gap_mask(module_size, gap_size, grid):
    """Gap mask of a detector made of `grid` = (rows, columns) modules of `module_size` = (h, w)
    pixels, separated by gaps of `gap_size` = (h, w) pixels. True = gap."""
    (mh, mw), (gh, gw), (nr, nc) = module_size, gap_size, grid
    mask = np.zeros((nr * mh + (nr - 1) * gh, nc * mw + (nc - 1) * gw), dtype=bool)
    for i in range(1, nr):
        mask[i * (mh + gh) - gh:i * (mh + gh)] = True
    for j in range(1, nc):
        mask[:, j * (mw + gw) - gw:j * (mw + gw)] = True
    return mask


def pilatus3_2m_gap_mask():
    """(1475, 1679) gap mask of the Pilatus3 2M. True = gap."""
    return detector_gap_mask(**PILATUS3_2M)


def stripe_gap_mask(shape, orientation, widths, offsets, even_widths=True):
    """Synthetic training gap: a horizontal stripe, a vertical stripe or a cross of both.

    Args:
        shape: (H, W) of the patch.
        orientation: "horizontal", "vertical" or "cross".
        widths: (height of the horizontal stripe, width of the vertical stripe) in pixels.
        offsets: (y, x) offset of the stripe centre from the patch centre.
        even_widths: if True, a stripe of width w covers 2 * (w // 2) pixels, i.e. odd widths
            lose one pixel. This is how the released model was trained; set False to get exactly
            w pixels.

    Returns:
        (H, W) bool mask, True = gap.
    """
    mask = np.zeros(shape, dtype=bool)

    def stripe(n, offset, width):
        centre = n // 2 + offset
        start = centre - width // 2
        stop = centre + width // 2 if even_widths else start + width
        return slice(start, stop)

    if orientation in ("horizontal", "cross"):
        mask[stripe(shape[0], offsets[0], widths[0]), :] = True
    if orientation in ("vertical", "cross"):
        mask[:, stripe(shape[1], offsets[1], widths[1])] = True
    if orientation not in ("horizontal", "vertical", "cross"):
        raise ValueError(f"unknown gap orientation {orientation!r}")
    return mask
