from typing import Any, Dict

import logging

import hydra
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def _flatten_dict(d, parent_key="", sep="/"):
    """Recursively flatten a nested dict: {"a": {"b": 1}} -> {"a/b": 1}."""
    items = []
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key, sep=sep).items())
        else:
            items.append((key, v))
    return dict(items)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Log all hyperparameters and parameter counts to the experiment logger.

    Flattens the full Hydra config so every setting is visible as a
    top-level key in wandb (e.g. ``force_field_module/net/rope_sigma``).
    """

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found; skipping hyperparameter logging.")
        return

    hparams = _flatten_dict(cfg)
    hparams.update(
        {
            "model/params/total": sum(p.numel() for p in model.parameters()),
            "model/params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model/params/non_trainable": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "output_dir": hydra.core.hydra_config.HydraConfig.get()["runtime"].get("output_dir"),
        }
    )

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
        if hasattr(logger, "log"):
            logger.log({"num_params": hparams["model/params/trainable"]})
