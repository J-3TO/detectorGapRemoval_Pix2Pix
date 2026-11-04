"""Generator and discriminator.

The U-Net building blocks (DoubleConv, Down, Up, OutConv) follow
https://github.com/milesial/Pytorch-UNet (GPL-3.0). Layer names are kept unchanged so that the
released weights load with ``load_state_dict``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interpolation import interpolate_gaps_torch


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool /2 -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Bilinear upsampling x2 -> concatenation with the skip connection -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if the input size is not a multiple of 16
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetMask(nn.Module):
    """Generator: a U-Net that gets the gap-interpolated image and the mask (1 = valid,
    0 = gap) as two input channels and predicts a correction that is added to the input inside
    the gap only, so pixels outside the gap are passed through unchanged:

        output = x + unet([x, mask]) * (1 - mask)

    `forward` is the training interface: x is already z-scored and interpolated (as the training
    patches are). For inference on raw images use `predict`, which does both itself.
    """

    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(2, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 1)

    def forward(self, x, mask):
        x1 = self.inc(torch.cat([x, mask], dim=1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        return x + self.outc(y) * (1 - mask)

    @torch.no_grad()
    def predict(self, x, valid):
        """Fill the gaps of raw images (inference; the model should be in eval mode):

            1. z-score each image with the mean / std of its valid pixels
            2. interpolate the gaps linearly
            3. forward
            4. undo the z-score; valid pixels are returned unchanged

        Args:
            x: (B, 1, H, W) images, e.g. flat-field corrected projections; values in the gaps
                (also NaN) are ignored.
            valid: (B, 1, H, W) or (1, 1, H, W), 1 = valid pixel, 0 = gap; the gaps must be
                complete rows and columns.
        """
        valid = valid.to(x.dtype).expand_as(x)
        gap = valid == 0
        n = valid.sum(dim=(-2, -1), keepdim=True)
        x0 = torch.where(gap, 0.0, x)
        mean = x0.sum(dim=(-2, -1), keepdim=True) / n
        var = (torch.where(gap, 0.0, x0 - mean) ** 2).sum(dim=(-2, -1), keepdim=True) / n
        std = var.sqrt().clamp(min=1e-12)
        y = self((interpolate_gaps_torch(x0, valid) - mean) / std, valid) * std + mean
        return torch.where(gap, y, x)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (Isola et al., 2017). It scores the pair (conditioning input,
    real or generated image) and returns one logit per overlapping image patch."""

    def __init__(self, in_channels=2, n_filters=64):
        super().__init__()

        def block(in_ch, out_ch, stride, use_norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not use_norm)]
            if use_norm:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, n_filters, stride=2, use_norm=False),
            *block(n_filters, n_filters * 2, stride=2),
            *block(n_filters * 2, n_filters * 4, stride=2),
            *block(n_filters * 4, n_filters * 8, stride=1),
            nn.Conv2d(n_filters * 8, 1, kernel_size=4, stride=1, padding=1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, input_img, target_img):
        return self.model(torch.cat([input_img, target_img], dim=1))
