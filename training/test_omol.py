"""Test-only entrypoint for OMol25.

Loads a checkpoint and runs `trainer.test()` on the full heldout val set
(no `limit_val_batches`). Mirrors `train_omol.py` but skips `trainer.fit()`.

Usage (Hydra overrides as for train_omol.py), e.g.:

    python test_omol.py \
        +checkpoint_path=/path/to/last.ckpt \
        force_field_module=platoformer \
        data=omol_4m \
        ...

Requires `+checkpoint_path=<path>` to be set.
"""
import logging
import os

import hydra
import torch

# Match train_omol.py: torch 2.11+ defaults `weights_only=True` which rejects
# our checkpoints (they hold OmegaConf objects). Patch torch.load globally.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

if hasattr(torch._dynamo.config, "cache_size_limit"):
    torch._dynamo.config.cache_size_limit = 128
elif hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 128

import lightning as L
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import OmegaConf
import rootutils

log = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def build_loggers(cfg):
    loggers = []
    if cfg.wandb.use_wandb:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            entity=cfg.wandb.entity,
            project=cfg.wandb.wandb_project,
            name=cfg.exp_name,
            group=cfg.wandb.group,
            dir=os.getenv("WANDB_DIR", "."),
        )
        loggers.append(wandb_logger)
    return loggers


@hydra.main(config_path="./configs/", config_name="train_omol", version_base="1.2")
def main(cfg):
    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    if os.environ.get("PSL_FAST_BACKENDS", "0") == "1":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        log.info("PSL_FAST_BACKENDS=1: cudnn.benchmark=True, deterministic=False, TF32 enabled")
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    L.seed_everything(cfg.seed)

    if not cfg.checkpoint_path:
        raise ValueError(
            "test_omol.py requires +checkpoint_path=<path-to-ckpt>; got empty checkpoint_path."
        )

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)

    log.info(f"Instantiating model <{cfg.force_field_module._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.force_field_module)

    # `omol_module.py` only initialises `self.free_scheduler` inside
    # configure_optimizers(), which Lightning skips for trainer.test(). Without
    # this, on_test_model_eval() crashes with AttributeError on `free_scheduler`.
    # Test-only path has no optimizer object, so False is correct.
    model.free_scheduler = False

    # The checkpoint was saved during fit() where the model's `setup(stage="fit")`
    # ran `self.net = torch.compile(self.net)`. That wrapper inserts `_orig_mod.`
    # prefixes into the saved state_dict. `setup(stage="test")` is a no-op for
    # compile, so we must compile manually here to make keys line up.
    if cfg.force_field_module.compile:
        log.info("Pre-compiling model.net so checkpoint state_dict keys (with _orig_mod.) match.")
        model.net = torch.compile(model.net)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=build_loggers(cfg))

    log.info(OmegaConf.to_yaml(cfg))

    # `trainer.test(dataloaders=...)` does NOT auto-call datamodule.setup(),
    # so self.datasets["val"] would be missing. Call setup() explicitly.
    datamodule.setup(stage="test")

    log.info(f"Loading checkpoint and running full-val test: {cfg.checkpoint_path}")
    trainer.test(model, dataloaders=datamodule.val_dataloader(), ckpt_path=cfg.checkpoint_path)


if __name__ == "__main__":
    main()
