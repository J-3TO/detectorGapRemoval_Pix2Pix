"""Loss terms of the generator besides the adversarial loss, and the histogram metric.

All take tensors of shape (B, C, H, W) and a gap tensor of the same shape (1 = gap, 0 = valid).

The histogram metric `histogram_loss` is adapted from the histogram loss of ADL
(https://github.com/mogvision/ADL), distributed under the MIT license:

    Copyright (c) 2022 mogvision

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch
import torch.nn.functional as F

# Gaps whose ground-truth std (in units of the z-scored patch) is below this are skipped by the
# EMD loss: there is no distribution to match, and for a std near float32 resolution
# mean +- n_sigma * std rounds to a single value, i.e. a zero bin width (division by zero -> NaN).
MIN_GAP_STD = 1e-4


def mse_loss(fake, real):
    """Mean squared error over the whole patch. Outside the gap the generator output equals the
    input, so this is the error in the gap, scaled by the gap fraction."""
    return F.mse_loss(fake, real)


def mean_loss(fake, real, gap):
    """Squared difference between the mean predicted and the mean true intensity, over the gap
    pixels of the whole batch."""
    n = gap.sum().clamp_min(1)
    return ((fake * gap).sum() / n - (real * gap).sum() / n) ** 2


def soft_histogram(values, lo, hi, n_bins=128, bandwidth=1.0):
    """Differentiable histogram of a 1D tensor over [lo, hi] (the bins of torch.histc).

    Every value is spread as a Gaussian of standard deviation `bandwidth` bin widths, and each
    bin gets the part of that Gaussian that falls inside it. A value well inside the range thus
    adds exactly 1 count in total, and for bandwidth -> 0 this becomes the hard histogram.
    """
    edges = torch.linspace(lo, hi, n_bins + 1, device=values.device, dtype=values.dtype)
    cdf = torch.special.ndtr((edges[None, :] - values[:, None]) / (bandwidth * (hi - lo) / n_bins))
    return (cdf[:, 1:] - cdf[:, :-1]).sum(dim=0)


def histogram_emd_loss(real, fake, gap, n_bins=128, n_sigma=3.0, bandwidth=1.0):
    """Earth Mover's Distance between the intensity distributions in the gap (training loss).

    Per image, the gap pixels of the ground truth and the prediction are binned into soft
    histograms over mean +- n_sigma * std of the ground-truth gap pixels and normalised by the
    number of gap pixels. The 1D EMD is the area between their cumulative distributions: how far,
    in intensity units, the predicted values have to be moved to match the ground truth, so every
    gap pixel is pulled towards the target distribution. Predictions beyond the bin range count
    as missing mass but get no gradient. Returns the mean over the images.
    """
    total, count = fake.new_zeros(()), 0
    for b in range(real.shape[0]):
        for c in range(real.shape[1]):
            sel = gap[b, c] > 0
            values = real[b, c][sel].detach()
            if values.numel() == 0:
                continue
            mean, std = values.mean(), values.std()
            if not (torch.isfinite(mean) and torch.isfinite(std)) or std < MIN_GAP_STD:
                continue
            lo, hi = (mean - n_sigma * std).item(), (mean + n_sigma * std).item()
            cdf_real = torch.cumsum(soft_histogram(values, lo, hi, n_bins, bandwidth), 0) / values.numel()
            cdf_fake = torch.cumsum(soft_histogram(fake[b, c][sel], lo, hi, n_bins, bandwidth), 0) / values.numel()
            total = total + (hi - lo) / n_bins * (cdf_fake - cdf_real).abs().sum()
            count += 1
    return total / max(count, 1)


@torch.no_grad()
def histogram_loss(real, fake, gap, n_bins=128, n_sigma=3.0):
    """Histogram metric of the gap pixels (HistoLoss, evaluation only).

    Per image, hard histograms (torch.histc) of the ground-truth and the predicted gap pixels over
    mean +- n_sigma * std of the ground truth, compared with the log-cosh loss and normalised by
    the number of tensor elements.
    """
    total = fake.new_zeros(())
    for b in range(real.shape[0]):
        for c in range(real.shape[1]):
            sel = gap[b, c] > 0
            values = real[b, c][sel]
            if values.numel() == 0:
                continue
            mean, std = values.mean(), values.std()
            if not (torch.isfinite(mean) and torch.isfinite(std)) or std == 0:
                continue
            lo, hi = (mean - n_sigma * std).item(), (mean + n_sigma * std).item()
            diff = torch.histc(values, bins=n_bins, min=lo, max=hi) - torch.histc(fake[b, c][sel], bins=n_bins, min=lo, max=hi)
            total = total + (diff + F.softplus(-2.0 * diff) - torch.log(torch.tensor(2.0, dtype=diff.dtype, device=diff.device))).mean()
    return total / real.numel()
