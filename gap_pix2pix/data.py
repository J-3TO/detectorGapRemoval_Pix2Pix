"""Training data: gap-free images -> (input with an interpolated synthetic gap, mask, target).

Three kinds of sources, all yielding z-scored 2D patches of the same size, so they can be mixed
in one batch (`RandomMix`):

* `ProjectionDataset`: .npy stacks (N, H, W) of gap-free images, e.g. the projections of one
  detector module. Each image is z-scored as a whole, then a random patch is cut.
* `SinogramDataset`: .npy volumes (detector rows, angles, detector columns), i.e. projection
  stacks with the angle axis in the middle, so that `volume[row]` is the sinogram of one detector
  row (a contiguous read). Patches are (angles x columns) windows inside one gap-free detector
  region ("module"); they are z-scored over the full module width of their angle window. Only
  vertical gaps are drawn, the only gaps a sinogram has.
* `ImageFolderDataset`: natural images (any format Pillow reads), converted to grayscale.

Every sample then goes through `make_sample`: optional augmentation (Poisson noise, Gaussian
blur, flips), a synthetic gap, linear interpolation across it.
"""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from .interpolation import interpolate_gaps
from .masks import PILATUS3_2M, stripe_gap_mask


def _expand(files, root=None):
    """A glob pattern or a list of paths/patterns (relative to `root`) -> sorted list of paths."""
    patterns = [files] if isinstance(files, str) else list(files)
    patterns = [os.path.join(root, p) for p in patterns] if root else patterns
    paths = sorted(p for pattern in patterns for p in (glob.glob(pattern) or [pattern]))
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing data files: {missing}")
    return paths


def zscore(img):
    return (img - img.mean()) / img.std()


# ---------------------------------------------------------------------------
# Augmentation and synthetic gaps (shared by all sources)
# ---------------------------------------------------------------------------

def random_poisson_noise(img, prob=0.6, counts=(5, 1000)):
    """With probability `prob`: rescale to [0, alpha] counts (alpha uniform in `counts`), draw
    Poisson noise, scale back."""
    if random.random() > prob:
        return img
    lo, hi = img.min(), img.max()
    if hi == lo:
        return img
    alpha = np.random.uniform(*counts)
    noisy = np.random.poisson(np.clip(alpha * (img - lo) / (hi - lo), 0, alpha)) / alpha
    return noisy * (hi - lo) + lo


def random_blur(img, prob=0.2, sigma=(0.5, 3.0)):
    if random.random() > prob:
        return img
    return gaussian_filter(img, sigma=random.uniform(*sigma))


def random_flip(img, prob=0.5):
    """With probability `prob`, flip each axis with probability 1/2."""
    if random.random() > prob:
        return img
    return img[::np.random.choice([1, -1]), ::np.random.choice([1, -1])].copy()


def random_gap_mask(shape, orientations, gap_width_y, gap_width_x, margin=5):
    """A random synthetic gap: orientation drawn from `orientations`, widths uniform in
    `gap_width_y` / `gap_width_x` (inclusive), position random but at least `margin` pixels from
    the patch border. True = gap."""
    orientation = random.choice(orientations)
    widths = (random.randint(*gap_width_y), random.randint(*gap_width_x))
    limits = (shape[0] // 2 - widths[0] - margin, shape[1] // 2 - widths[1] - margin)
    offsets = tuple(int(random.uniform(-limit, limit)) for limit in limits)
    return stripe_gap_mask(shape, orientation, widths, offsets)


def make_sample(target, orientations=("horizontal", "vertical", "cross"), gap_width_y=(7, 40),
                gap_width_x=(7, 40), augment=False):
    """Training triple from a z-scored, gap-free patch.

    Returns float32 arrays of shape (1, H, W):
        input:  the patch with a synthetic gap, linearly interpolated across the gap
        valid:  1 = valid pixel, 0 = gap
        target: the (augmented) gap-free patch
    """
    if augment:
        target = random_flip(random_blur(random_poisson_noise(target)))
    gap = random_gap_mask(target.shape, orientations, gap_width_y, gap_width_x)
    inp = interpolate_gaps(target, gap)
    return (inp[None].astype(np.float32), (~gap)[None].astype(np.float32),
            target[None].astype(np.float32))


class _PatchSource(torch.utils.data.Dataset):
    """Common part: patches are drawn at random (the index only fixes the epoch length)."""

    def __init__(self, patch_size, gap_orientations, gap_width_y, gap_width_x, augment, length):
        self.patch_size = tuple(patch_size)
        self.gap = dict(orientations=tuple(gap_orientations), gap_width_y=tuple(gap_width_y),
                        gap_width_x=tuple(gap_width_x), augment=augment)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return make_sample(self.load_patch(), **self.gap)

    def load_patch(self):
        raise NotImplementedError

    def random_crop(self, img):
        y = random.randrange(img.shape[0] - self.patch_size[0] + 1)
        x = random.randrange(img.shape[1] - self.patch_size[1] + 1)
        return img[y:y + self.patch_size[0], x:x + self.patch_size[1]]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class ProjectionDataset(_PatchSource):
    """Patches of gap-free images stored as .npy stacks (N, H, W).

    Args:
        files: glob pattern or list of .npy files.
        root: optional directory that `files` are relative to.
        patch_size: (h, w) of the patches; images must be at least this large.
        gap_orientations, gap_width_y, gap_width_x: synthetic gaps, see `random_gap_mask`.
        augment: Poisson noise, blur and flips.
        length: samples per epoch; default the total number of images.
    """

    def __init__(self, files, root=None, patch_size=(256, 128),
                 gap_orientations=("horizontal", "vertical", "cross"), gap_width_y=(7, 40),
                 gap_width_x=(7, 40), augment=False, length=None):
        self.files = _expand(files, root)
        self.sizes = [np.load(f, mmap_mode="r").shape[0] for f in self.files]
        super().__init__(patch_size, gap_orientations, gap_width_y, gap_width_x, augment,
                         length or sum(self.sizes))
        self._stacks = {}  # opened lazily, so that every DataLoader worker maps its own files

    def load_patch(self):
        i = random.choices(range(len(self.files)), weights=self.sizes)[0]
        if i not in self._stacks:
            self._stacks[i] = np.load(self.files[i], mmap_mode="r")
        img = np.asarray(self._stacks[i][random.randrange(self.sizes[i])], dtype=np.float64)
        return self.random_crop(zscore(img))


class ImageFolderDataset(_PatchSource):
    """Patches of natural images (e.g. the ImageNet validation set), converted to grayscale.
    Images not larger than the patch size are skipped.

    Args:
        root: image directory.
        file_list: optional text file with one path (relative to `root`) per line; default all
            .jpg/.jpeg/.png/.tif files below `root`.
        Other arguments as for `ProjectionDataset`; `length` defaults to the number of images.
    """

    EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    def __init__(self, root, file_list=None, patch_size=(256, 128),
                 gap_orientations=("horizontal", "vertical", "cross"), gap_width_y=(7, 40),
                 gap_width_x=(7, 40), augment=False, length=None):
        if file_list:
            with open(file_list) as f:
                self.files = [os.path.join(root, line.strip()) for line in f if line.strip()]
        else:
            self.files = sorted(os.path.join(d, n) for d, _, names in os.walk(root)
                                for n in names if n.lower().endswith(self.EXTENSIONS))
        if not self.files:
            raise FileNotFoundError(f"no images in {root}")
        super().__init__(patch_size, gap_orientations, gap_width_y, gap_width_x, augment,
                         length or len(self.files))

    def load_patch(self):
        for _ in range(150):
            img = np.asarray(Image.open(random.choice(self.files)).convert("L"), dtype=np.float64)
            if img.shape[0] > self.patch_size[0] and img.shape[1] > self.patch_size[1] and img.std() > 0:
                return self.random_crop(zscore(img))
        raise RuntimeError(f"no image larger than {self.patch_size} found in 150 draws")


class SinogramDataset(_PatchSource):
    """(angles x columns) patches of sinograms, inside one gap-free detector region.

    Each volume (detector rows, angles, detector columns) is split into `modules`, gap-free boxes
    (row0, row1, col0, col1) of the detector; real detector gaps are therefore never seen, whatever
    they contain. A sample is a volume (weighted by its number of angles), a module, a detector row
    in the module and a window of patch_size[0] angles. The module width x angle window is
    z-scored, then patch_size[1] columns are cut from it. All volumes and modules are used; only
    constant windows, which cannot be z-scored, are redrawn. The z-score makes the values'
    scale irrelevant, so e.g. linearly quantized uint16 volumes can be used as stored.

    Args:
        files: glob pattern or list of .npy volumes.
        root: optional directory that `files` are relative to.
        modules: list of (row0, row1, col0, col1); "pilatus3_2m" for the 24 modules of that
            detector; default (None) the whole detector is one module.
        gap_orientations: default vertical only.
        Other arguments as for `ProjectionDataset`; `length` defaults to the total number of angles.
    """

    def __init__(self, files, root=None, modules=None, patch_size=(256, 128),
                 gap_orientations=("vertical",), gap_width_y=(7, 40), gap_width_x=(7, 40),
                 augment=False, length=None):
        self.volumes = []  # (path, n_angles, modules)
        for path in _expand(files, root):
            n_rows, n_angles, n_cols = np.load(path, mmap_mode="r").shape
            boxes = self._modules(modules, n_rows, n_cols)
            if n_angles < patch_size[0]:
                raise ValueError(f"{path}: {n_angles} angles, fewer than the patch height {patch_size[0]}")
            if any(c1 - c0 < patch_size[1] for _, _, c0, c1 in boxes):
                raise ValueError(f"patch width {patch_size[1]} exceeds a module width")
            self.volumes.append((path, n_angles, boxes))
        self.weights = [n for _, n, _ in self.volumes]
        super().__init__(patch_size, gap_orientations, gap_width_y, gap_width_x, augment,
                         length or sum(self.weights))
        self._mmaps = {}

    @staticmethod
    def _modules(modules, n_rows, n_cols):
        if modules is None:
            return [(0, n_rows, 0, n_cols)]
        if modules == "pilatus3_2m":
            (mh, mw), (gh, gw), (nr, nc) = (PILATUS3_2M[k] for k in ("module_size", "gap_size", "grid"))
            return [(i * (mh + gh), i * (mh + gh) + mh, j * (mw + gw), j * (mw + gw) + mw)
                    for i in range(nr) for j in range(nc)]
        return [tuple(m) for m in modules]

    def load_patch(self):
        n_angles = self.patch_size[0]
        for _ in range(100):
            path, total_angles, boxes = random.choices(self.volumes, weights=self.weights)[0]
            if path not in self._mmaps:
                self._mmaps[path] = np.load(path, mmap_mode="r")
            r0, r1, c0, c1 = random.choice(boxes)
            a = random.randrange(total_angles - n_angles + 1)
            window = np.asarray(self._mmaps[path][random.randrange(r0, r1), a:a + n_angles, c0:c1],
                                dtype=np.float64)
            if window.std() == 0:  # constant window: cannot be z-scored
                continue
            c = random.randrange(window.shape[1] - self.patch_size[1] + 1)
            return (window[:, c:c + self.patch_size[1]] - window.mean()) / window.std()
        raise RuntimeError("drew 100 constant sinogram windows in a row")


class RandomMix(torch.utils.data.Dataset):
    """Draws every sample from one of several sources, chosen at random with the given weights."""

    def __init__(self, sources, weights, length):
        if len({s.patch_size for s in sources}) != 1:
            raise ValueError("all sources must have the same patch_size")
        self.sources, self.weights, self.length = sources, weights, int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        source = random.choices(self.sources, weights=self.weights)[0]
        return source[idx % len(source)]


SOURCES = {"projections": ProjectionDataset, "sinograms": SinogramDataset, "images": ImageFolderDataset}


def build_dataset(sources, length, patch_size, gap_width_y, gap_width_x, augment):
    """A `RandomMix` from the `sources` list of a training config (see config.yml): every entry
    has a `type` (projections | sinograms | images), a `weight` and the arguments of that class."""
    datasets, weights = [], []
    for spec in sources:
        spec = dict(spec)
        cls, weight = SOURCES[spec.pop("type")], spec.pop("weight")
        if weight <= 0:
            continue
        spec = dict(dict(patch_size=patch_size, gap_width_y=gap_width_y, gap_width_x=gap_width_x,
                         augment=augment), **spec)
        datasets.append(cls(**spec))
        weights.append(weight)
    return RandomMix(datasets, weights, length)
