"""Pix2Pix training (Lightning module).

Each step updates the discriminator, then the generator:

    D: BCE(D(input, target), 0.9) + BCE(D(input, G(input)), 0)          (x 0.5)
    G: lambda_adv  * BCE(D(input, G(input)), 0.9)
     + lambda_mse  * MSE(G(input), target)
     + lambda_mean * (mean of G(input) in the gap - mean of target in the gap)^2
     + lambda_histo * EMD between the soft histograms of the gap pixels

`input` is the linearly interpolated patch; the discriminator sees (input, output) pairs.
"""
import os

import lightning
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .losses import histogram_emd_loss, histogram_loss, mean_loss, mse_loss
from .models import PatchGANDiscriminator, UNetMask


class Pix2PixGap(lightning.LightningModule):
    """
    Args:
        lr, betas: Adam settings of both networks.
        lambda_adv, lambda_mse, lambda_mean, lambda_histo: generator loss weights
            (lambda_histo = 0 switches the histogram loss off).
        label_smooth: target of the discriminator for real pairs (one-sided label smoothing).
        histo_bandwidth: kernel width of the soft histograms, in bins.

    The defaults are the settings of the last training stage of the released model. Validation
    logs the MSE in the gap (`val_gap_mse`), the gap-mean loss, the hard histogram metric
    (`val_histo_loss`) and the EMD (`val_histo_emd`).
    """

    def __init__(self, lr=2e-4, betas=(0.5, 0.999), lambda_adv=2.0, lambda_mse=1.0,
                 lambda_mean=100.0, lambda_histo=3.0, label_smooth=0.9, histo_bandwidth=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.generator = UNetMask()
        self.discriminator = PatchGANDiscriminator()
        self.automatic_optimization = False  # alternating D / G updates
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x, valid):
        return self.generator(x, valid)

    def _adversarial(self, logits, real):
        target = torch.full_like(logits, self.hparams.label_smooth if real else 0.0)
        return self.bce(logits, target)

    def _generator_losses(self, inp, fake, real, gap):
        h = self.hparams
        losses = {
            "adv_loss": self._adversarial(self.discriminator(inp, fake), real=True),
            "mse_loss": mse_loss(fake, real),
            "mean_loss": mean_loss(fake, real, gap),
            # without a weight, computed on a detached copy: logged only, no backward cost
            "histo_loss": histogram_emd_loss(real, fake if h.lambda_histo else fake.detach(), gap,
                                             bandwidth=h.histo_bandwidth),
        }
        losses["g_loss"] = (h.lambda_adv * losses["adv_loss"] + h.lambda_mse * losses["mse_loss"]
                            + h.lambda_mean * losses["mean_loss"] + h.lambda_histo * losses["histo_loss"])
        return losses

    def _is_finite(self, loss, name):
        """False (and a logged `nonfinite_steps` count) for a NaN/inf loss: the optimizer step is
        then skipped, so a single bad batch cannot write NaN into the weights."""
        if torch.isfinite(loss):
            return True
        print(f"WARNING: non-finite {name} at epoch {self.current_epoch}, step {self.global_step}; step skipped")
        self.log("nonfinite_steps", 1.0, on_step=False, on_epoch=True, reduce_fx="sum")
        return False

    def training_step(self, batch, batch_idx):
        inp, valid, real = batch
        gap = 1 - valid
        opt_g, opt_d = self.optimizers()

        # discriminator
        self.toggle_optimizer(opt_d)
        fake = self(inp, valid)
        d_loss = 0.5 * (self._adversarial(self.discriminator(inp, real), real=True)
                        + self._adversarial(self.discriminator(inp, fake.detach()), real=False))
        if self._is_finite(d_loss, "d_loss"):
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()
        self.untoggle_optimizer(opt_d)

        # generator
        self.toggle_optimizer(opt_g)
        fake = self(inp, valid)
        losses = self._generator_losses(inp, fake, real, gap)
        if self._is_finite(losses["g_loss"], "g_loss"):
            opt_g.zero_grad()
            self.manual_backward(losses["g_loss"])
            opt_g.step()
        self.untoggle_optimizer(opt_g)

        losses["d_loss"] = d_loss
        self.log_dict({k: v.detach() for k, v in losses.items()}, on_epoch=True, prog_bar=True,
                      sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inp, valid, real = batch
        gap = 1 - valid
        fake = self(inp, valid)
        self.log_dict({
            "val_loss": mse_loss(fake, real),
            "val_gap_mse": ((fake - real) ** 2 * gap).sum() / gap.sum().clamp_min(1),
            "val_mean_loss": mean_loss(fake, real, gap),
            "val_histo_loss": histogram_loss(real, fake, gap),
            "val_histo_emd": histogram_emd_loss(real, fake, gap),
        }, on_epoch=True, prog_bar=True, sync_dist=True)
        if batch_idx == 0:
            self._preview = (inp[:4].cpu(), fake[:4].cpu(), real[:4].cpu())

    def on_validation_epoch_end(self):
        """Saves <log_dir>/previews/epoch_XXXX.png: input | prediction | target of 4 patches."""
        if not self.trainer.is_global_zero or self.trainer.sanity_checking or not hasattr(self, "_preview"):
            return
        rows = []
        for inp, fake, real in zip(*self._preview):
            lo, hi = (real.mean() - 2 * real.std()).item(), (real.mean() + 2 * real.std()).item()
            rows.append(np.concatenate([np.clip((t[0].numpy() - lo) / (hi - lo), 0, 1)
                                        for t in (inp, fake, real)], axis=1))
        out_dir = os.path.join(self.trainer.log_dir or ".", "previews")
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray((np.concatenate(rows, axis=0) * 255).astype(np.uint8)).save(
            os.path.join(out_dir, f"epoch_{self.current_epoch:04d}.png"))

    def configure_optimizers(self):
        h = self.hparams
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=h.lr, betas=tuple(h.betas))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=h.lr, betas=tuple(h.betas))
        return [opt_g, opt_d]
