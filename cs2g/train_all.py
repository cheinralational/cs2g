import os, sys
import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

_base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _base_dir)

from utils.util import instantiate_from_config
import yaml

_AUTO_ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"

DATASET_CONFIGS = {
    "CVUSA": {
        "s1": "configs/Boost_Sat2Den/CVUSA_geo_ldm.yaml",
        "s2": "configs/Boost_Sat2Den/CVUSA_geo_ldm.yaml",
    },
    "KITTI": {
        "s1": "configs/Boost_Sat2Den/KITTI_geo_ldm.yaml",
        "s2": "configs/Boost_Sat2Den/KITTI_geo_ldm.yaml",
    },
    "VIGOR": {
        "s1": "configs/Boost_Sat2Den/VIGOR_geo_ldm.yaml",
        "s2": "configs/Boost_Sat2Den/VIGOR_geo_ldm.yaml",
    },
    "crossgeo": {
        "s1": "configs/Boost_Sat2Den/crossgeo_geo_ldm_stage1.yaml",
        "s2": "configs/Boost_Sat2Den/crossgeo_geo_ldm_sphere.yaml",
    },
}


def train_stage1(config_path, logdir="outputs", devices="0",
                 max_epochs=50, batch_size=4, accumulate_grad_batches=4,
                 lr=None, resume=None, seed=24):
    seed_everything(seed)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data = instantiate_from_config(config["data"])
    data.batch_size = batch_size
    data.num_workers = 0

    model = instantiate_from_config(config["model"])
    model.use_spherical_control = False

    if lr is not None:
        model.learning_rate = lr

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir_full = os.path.join(logdir, "stage1", now)
    ckpt_dir = os.path.join(logdir_full, "checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}",
        save_top_k=1,
        verbose=True,
        monitor="L1_loss",
        mode="min",
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=_AUTO_ACCELERATOR,
        devices=devices if _AUTO_ACCELERATOR == "gpu" else 1,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="16-mixed" if _AUTO_ACCELERATOR == "gpu" else "32",
        default_root_dir=logdir_full,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=5,
    )

    if resume:
        trainer.fit(model, data, ckpt_path=resume)
    else:
        trainer.fit(model, data)

    print(f"Stage-1 training complete: {logdir_full}")
    return logdir_full


def train_stage2(config_path, logdir="outputs", devices="0",
                 max_epochs=50, batch_size=2, accumulate_grad_batches=4,
                 lr=None, stage1_ckpt=None, seed=24):
    seed_everything(seed)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data = instantiate_from_config(config["data"])
    data.batch_size = batch_size
    data.num_workers = 0

    model = instantiate_from_config(config["model"])
    model.use_spherical_control = True

    if lr is not None:
        model.learning_rate = lr

    if stage1_ckpt is not None and os.path.exists(stage1_ckpt):
        print(f"Loading Stage-1 checkpoint: {stage1_ckpt}")
        checkpoint = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        if "DDPM.denoise_model" in name:
            param.requires_grad = False
        if "condition_model_sat" in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params / 1e6:.2f}M / Total: {total_params / 1e6:.2f}M")

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir_full = os.path.join(logdir, "stage2", now)
    ckpt_dir = os.path.join(logdir_full, "checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}",
        save_top_k=1,
        verbose=True,
        monitor="L1_loss",
        mode="min",
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=_AUTO_ACCELERATOR,
        devices=devices if _AUTO_ACCELERATOR == "gpu" else 1,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="16-mixed" if _AUTO_ACCELERATOR == "gpu" else "32",
        default_root_dir=logdir_full,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=5,
    )

    trainer.fit(model, data)
    print(f"Stage-2 training complete: {logdir_full}")
    return logdir_full


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="crossgeo", choices=["CVUSA", "KITTI", "VIGOR", "crossgeo"])
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--logdir", type=str, default="outputs")
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--stage1-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()

    configs = DATASET_CONFIGS[args.dataset]
    stage_key = f"s{args.stage}"
    config_path = os.path.join(_base_dir, configs[stage_key])

    print(f"Dataset: {args.dataset} | Stage: {args.stage} | Config: {config_path}")
    print(f"Logdir: {args.logdir}")

    if args.stage == 1:
        train_stage1(
            config_path,
            logdir=args.logdir,
            devices=args.devices,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            accumulate_grad_batches=args.accumulate_grad_batches,
            lr=args.lr,
            resume=args.resume,
            seed=args.seed,
        )
    else:
        train_stage2(
            config_path,
            logdir=args.logdir,
            devices=args.devices,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            accumulate_grad_batches=args.accumulate_grad_batches,
            lr=args.lr,
            stage1_ckpt=args.stage1_ckpt,
            seed=args.seed,
        )
