"""Loading weights.

Two file formats are used:

* generator weights (`*_generator.pth`): the `state_dict` of `UNetMask`; all that inference needs.
* full weights (`*_full.pth`): {"generator": state_dict, "discriminator": state_dict,
  "epoch": int}; the starting point for continued training (with fresh optimizers).

Lightning checkpoints (`*.ckpt`, written during training) can be used wherever a full weights
file is expected.
"""
import os

import torch

from .models import UNetMask

# Hugging Face Hub repository with the released weights
HF_REPO_ID = "J-3TO/detectorGapRemoval_Pix2Pix"
HF_GENERATOR_FILE = "pix2pix_projections_sinograms_generator.pth"
HF_FULL_FILE = "pix2pix_projections_sinograms_full.pth"
HF_REVISION = "v1.0"  # tag of the released weights on the Hub


def download(filename=HF_GENERATOR_FILE, repo_id=HF_REPO_ID, revision=HF_REVISION):
    """Path of a released weights file, downloaded from the Hugging Face Hub on first use."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _resolve(path):
    """A local file, or "hf" / "hf:<filename>" for a released file on the Hugging Face Hub."""
    if path == "hf":
        return download()
    if path.startswith("hf:"):
        return download(path[3:])
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def load_state_dicts(path):
    """(generator state_dict, discriminator state_dict or None, epoch or None) of any weights
    file: generator-only .pth, full .pth or Lightning .ckpt."""
    path = _resolve(path)
    if path.endswith(".ckpt"):
        # Lightning checkpoints also contain trainer state (pickled Python objects): only load
        # checkpoints you trust.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        split = {net: {k[len(net) + 1:]: v for k, v in state.items() if k.startswith(net + ".")}
                 for net in ("generator", "discriminator")}
        return split["generator"], split["discriminator"], ckpt.get("epoch")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if "generator" in obj:
        return obj["generator"], obj.get("discriminator"), obj.get("epoch")
    return obj, None, None


def load_generator(path="hf", device="cpu"):
    """`UNetMask` in eval mode with the weights of `path` (any format, see `load_state_dicts`)."""
    generator = UNetMask()
    generator.load_state_dict(load_state_dicts(path)[0])
    return generator.to(device).eval()
