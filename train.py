"""Train the Pix2Pix gap-inpainting model; all settings are in the config file.

    python3 train.py --config config.yml

In the config, `init_weights` selects the weights to start from (fresh optimizers; empty = from
scratch) and `resume` a checkpoint of an interrupted run to continue exactly where it stopped
(weights, optimizers, epoch).
"""
import argparse
import os

import lightning
import torch
import yaml
from torch.utils.data import DataLoader

from gap_pix2pix.data import build_dataset
from gap_pix2pix.lightning_module import Pix2PixGap
from gap_pix2pix.weights import load_state_dicts


def make_loader(cfg, split, shuffle):
    dataset = build_dataset(patch_size=cfg["patch_size"], gap_width_y=cfg["gap_width_y"],
                            gap_width_x=cfg["gap_width_x"], **cfg["data"][split])
    return DataLoader(dataset, shuffle=shuffle, **cfg["loader"])


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    lightning.seed_everything(cfg.get("seed", 0), workers=True)
    torch.set_float32_matmul_precision("medium")
    train_loader = make_loader(cfg, "train", shuffle=True)
    val_loader = make_loader(cfg, "val", shuffle=False)

    model = Pix2PixGap(**cfg["model"])
    init, resume = cfg.get("init_weights"), cfg.get("resume")
    if init and resume:
        print(f"resume is set: continuing {resume}; init_weights ({init}) is ignored")
    elif init:
        generator, discriminator, _ = load_state_dicts(init)
        model.generator.load_state_dict(generator)
        if discriminator is not None:
            model.discriminator.load_state_dict(discriminator)
        print(f"initialised from {init} ({'generator + discriminator' if discriminator else 'generator only'})")

    logger = lightning.pytorch.loggers.CSVLogger(cfg.get("output_dir", "runs"), name=cfg["name"])
    checkpoints = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="{epoch}_{val_loss:.4f}",
        every_n_epochs=cfg.get("checkpoint_every_n_epochs", 50),
        save_top_k=-1,
        save_last=True,
    )
    trainer = lightning.Trainer(logger=logger, callbacks=[checkpoints], **cfg["trainer"])
    if trainer.is_global_zero:
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(os.path.join(logger.log_dir, "config.yml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="training config (yaml), e.g. config.yml")
    main(parser.parse_args().config)
