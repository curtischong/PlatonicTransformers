import os
import logging

import hydra
import torch
# Fix for torch 2.11+: checkpoints saved with torch 2.8 contain OmegaConf objects
# that weights_only=True rejects. Safe since these are our own checkpoints.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Raise dynamo recompile limit so variable-atom batches (dynamic_batching=true)
# don't hit the default 8-shape cap. With automatic_dynamic_shapes, a handful
# of concrete shapes are traced before PyTorch switches to fully dynamic; 128
# gives headroom for val-boundary shape changes as well.
# in some PyTorch versions this is cache_size_limit instead of recompile_limit.
if hasattr(torch._dynamo.config, 'cache_size_limit'):
    torch._dynamo.config.cache_size_limit = 128
elif hasattr(torch._dynamo.config, 'recompile_limit'):
    torch._dynamo.config.recompile_limit = 128

import lightning as L
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from src.utils.callbacks import EMACallback

import humanize
import rootutils
from utils.log import log_hyperparameters

log = logging.getLogger(__name__)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def default_checkpoint_root() -> str:
    """Prefer env override, otherwise use repo-local `checkpoints/`."""
    return os.getenv("CHECKPOINT_ROOT", os.path.join(rootutils.find_root(".project-root"), "checkpoints"))


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

    if cfg.comet.use_comet:
        from lightning.pytorch.loggers import CometLogger

        comet_logger = CometLogger(project=cfg.comet.group, name=cfg.exp_name)
        loggers.append(comet_logger)

    return loggers


def build_callbacks(cfg, model):
    run_id = os.getenv("SLURM_JOB_ID", "local")
    run_name = f"run_{run_id}_params_{humanize.intword(model.total_params)}"

    checkpoint_dir = os.path.join(default_checkpoint_root(), cfg.exp_name, cfg.model_name, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}",
        monitor="loss/val",
        save_last=True,
        mode="min",
        verbose=True,
    )

    callbacks = [checkpoint_callback]

    ema_cfg = cfg.get("ema", None)
    if ema_cfg is not None and ema_cfg.get("decay", None) is not None:
        decay = float(ema_cfg.decay)
        warmup_steps = int(ema_cfg.get("warmup_steps", 2000))
        log.info(f"EMA enabled: decay={decay}, warmup_steps={warmup_steps}")
        callbacks.append(EMACallback(decay=decay, warmup_steps=warmup_steps))

    return callbacks


def train(cfg):
    loggers = build_loggers(cfg)

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)

    log.info(f"Instantiating model <{cfg.force_field_module._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.force_field_module)

    callbacks = build_callbacks(cfg, model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log_hyperparameters({"cfg": cfg, "datamodule": datamodule, "model": model, "callbacks": callbacks, "trainer": trainer})

    log.info(OmegaConf.to_yaml(cfg))
    trainer.fit(model, datamodule, ckpt_path=cfg.checkpoint_path)
    trainer.test(model, dataloaders=datamodule.val_dataloader())


@hydra.main(config_path="./configs/", config_name="train_omol", version_base="1.2")
def main(cfg):
    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    # matmul + cudnn backend knobs (cfg-driven; defaults in train_omol.yaml)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(cfg.cudnn_deterministic)
    log.info(
        "matmul_precision=%s  cudnn.benchmark=%s  cudnn.deterministic=%s",
        cfg.matmul_precision, cfg.cudnn_benchmark, cfg.cudnn_deterministic,
    )

    # Dynamo tuning, applied automatically when compile is on. Without these,
    # the diagnostic in job 22569081 showed function 6 (PlatonicLinear.forward)
    # hitting 11 recompiles past the default cache_size_limit=8 and falling back
    # to eager — i.e. compile on the books but not in fact. These knobs are
    # intrinsic to making compile work on this model + dyn-batching, not optional.
    if cfg.force_field_module.compile:
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = 256
        _dynamo.config.force_parameter_static_shapes = False
        _dynamo.config.capture_scalar_outputs = True
        log.info(
            "compile=true → dynamo.config cache_size_limit=256, "
            "force_parameter_static_shapes=False, capture_scalar_outputs=True"
        )

    L.seed_everything(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    main()
