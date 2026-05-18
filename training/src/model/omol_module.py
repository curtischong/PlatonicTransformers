import logging
import time
import numpy as np
import torch
import lightning as L
from torch.nn import ModuleDict
from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from fairchem.core.modules.loss import DDPLoss
from src.utils.file_utils import load_reference
from nets.mup import mup_init, get_mup_multipliers, build_optimizer_param_groups
import schedulefree


log = logging.getLogger(__name__)


class GraphModel(L.LightningModule):
    def __init__(
        self,
        net,
        optimizer,
        train_augmentation,
        reference_path,
        compile,
        train_mean=None,
        train_rmsd=None,
        num_batches=None,
        base_net=None,
        flops_coef=6,
        compile_mode="default",
        compile_dynamic=None,
        skip_loss_above=0.0,
    ):
        super().__init__()

        # muP scaling
        self.base_net = base_net
        if base_net is not None:
            self.mup_multipliers = get_mup_multipliers(base_net, net)
            mup_init(net, self.mup_multipliers)
        else:
            self.mup_multipliers = None

        self.net = net
        self.optimizer = optimizer
        self.flops_coef = flops_coef

        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        self.total_params = getattr(self.net, "num_params", sum(np.prod(p.size()) for p in model_parameters))

        self.train_augmentation = train_augmentation
        # If train_augmentation is a string like "o3", include reflections; otherwise SO(3) only
        if isinstance(train_augmentation, str) and train_augmentation.lower() == "o3":
            self.train_augmentation = True
            self._augment_reflect = True
        else:
            self._augment_reflect = False
        self.train_mean = train_mean
        self.train_rmsd = train_rmsd
        self.num_batches = num_batches  # to estimate train mean and rmsd on the fly
        self.skip_loss_above = float(skip_loss_above or 0.0)

        log.info(f"Total params: {self.total_params}")

        self.save_hyperparameters(logger=False, ignore=["net", "base_net"])

        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "f_mae": MeanMetric(),
                "e_mae": MeanMetric(),
                "e_mae_per_atom": MeanMetric(),
                "f_loss": MeanMetric(),
                "e_loss": MeanMetric(),
                "rot_loss": MeanMetric(),
                "total": MeanMetric(),
            }
        )

        self.val_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "f_mae": MeanMetric(),
                "e_mae": MeanMetric(),
                "e_mae_per_atom": MeanMetric(),
                "f_loss": MeanMetric(),
                "e_loss": MeanMetric(),
                "total": MeanMetric(),
            }
        )

        self.test_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "f_mae": MeanMetric(),
                "e_mae": MeanMetric(),
                "e_mae_per_atom": MeanMetric(),
                "f_loss": MeanMetric(),
                "e_loss": MeanMetric(),
                "total": MeanMetric(),
            }
        )

        self.e_weight = self.hparams.optimizer.e_weight
        self.f_weight = self.hparams.optimizer.f_weight
        self.e_weight_warmup_steps = int(getattr(self.hparams.optimizer, "e_weight_warmup_steps", 0))

        self.element_references = load_reference(reference_path)

        self.e_loss_fn = DDPLoss(loss_name=self.hparams.optimizer.e_loss_name, reduction="mean")
        self.f_loss_fn = DDPLoss(loss_name=self.hparams.optimizer.f_loss_name, reduction="mean")

        self.token_processed = 0
        self.total_flops_used = 0
        # Cumulative wall-clock seconds spent inside train_batch_start→end.
        # Training-only time (excludes stage, compile, val/test, idle gaps) — the
        # right axis for scaling-law wall-clock comparisons at matched compute.
        self.train_seconds = 0.0
        self._train_step_start = None

    def on_save_checkpoint(self, checkpoint):
        # token_processed / total_flops_used / train_seconds are plain python attrs
        # (not buffers), so they are not in state_dict. Persist them explicitly so a
        # resumed run continues the FLOPs/tokens/wall-clock axes without restarting
        # from zero.
        checkpoint["token_processed"] = int(self.token_processed)
        checkpoint["total_flops_used"] = int(self.total_flops_used)
        checkpoint["train_seconds"] = float(self.train_seconds)

    def on_load_checkpoint(self, checkpoint):
        # token_processed and total_flops_used are logged with reduce_fx="sum"
        # over ranks; if N ranks each load the saved value, the post-resume
        # logged sum is N× the true offset. Divide on load so that the across-
        # rank sum at log time recovers the correct value.
        # train_seconds uses reduce_fx="mean" so no division is needed.
        try:
            import torch.distributed as dist
            ws = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        except Exception:
            ws = 1
        self.token_processed = int(checkpoint.get("token_processed", 0)) // ws
        self.total_flops_used = int(checkpoint.get("total_flops_used", 0)) // ws
        self.train_seconds = float(checkpoint.get("train_seconds", 0.0))

    def on_train_batch_start(self, batch, batch_idx):
        self._train_step_start = time.monotonic()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._train_step_start is not None:
            self.train_seconds += time.monotonic() - self._train_step_start
            self._train_step_start = None

    @torch.no_grad()
    def compute_stats(self, dataset=None):
        """Estimate normalization stats over a subset of training batches.

        Returns (train_mean, train_rmsd) where train_mean is fixed at 0.0 and
        train_rmsd is the RMS-from-zero of (energy - element_refs). This matches
        eSEN/UMA's Normalizer convention (mean=0, rmsd=fairchem.core.modules.
        normalization.normalizer.Normalizer): we trust the per-element references
        to absorb the per-atom mean of the energy distribution, so the residual
        is centered at zero by construction.
        """

        from torch.utils.data import DataLoader
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

        if dataset is None:
            dataset = self.trainer.datamodule.datasets["train"]

        dataloader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=atomicdata_list_to_batch,
        )
        num_batches = len(dataloader) if self.num_batches is None else self.num_batches

        all_ys = []
        for i, batch in tqdm(enumerate(dataloader), total=num_batches, desc="Estimating mean/rms"):
            if i == num_batches:
                break
            batch = batch.to(self.element_references.element_references.device)
            if hasattr(batch, "energy_ref_corrected"):
                energy = batch.energy_ref_corrected
            else:
                energy = self.element_references.apply_refs(batch, batch.energy)
            all_ys.append(energy)

        all_ys = torch.cat(all_ys, dim=0)
        # Diagnostic: residual mean after element refs (should be small if refs fit well)
        residual_mean = torch.mean(all_ys, dim=0).item()
        # eSEN-style RMS-from-zero (no mean subtracted): sqrt(mean(y^2))
        rms = ((all_ys ** 2).mean()).sqrt().item()

        # Don't call self.log() here — this method may be called from
        # on_fit_start, where Lightning forbids self.log(). Print to stdout
        # for visibility; deferred logging to wandb happens in on_train_start.
        log.info("compute_stats: residual_mean_after_refs=%.4f", residual_mean)
        log.info("compute_stats: train rms (eSEN, mean=0)=%.4f", rms)
        self._stats_residual_mean = residual_mean
        self._stats_rms = rms
        return 0.0, rms

    def forward(self, data):
        e_gt, f_gt = data.energy, data.forces
        preds = self.net(data)
        e_pred, f_pred = preds["energy"], preds["forces"]
        return e_pred.view(-1), e_gt.view(-1), f_pred, f_gt

    def on_fit_start(self):
        # Compute normalization stats BEFORE Lightning's sanity-check val pass,
        # which fires after on_fit_start but before on_train_start. If we wait
        # until on_train_start, validation_step hits train_rmsd=None and crashes.
        if self.train_mean is None or self.train_rmsd is None:
            self.train_mean, self.train_rmsd = self.compute_stats()

    def on_train_start(self):
        # Ensure ScheduleFree optimizer is in train mode (critical for checkpoint resumption)
        self.set_optimizer_state("train")
        self.log("total_parameters", self.total_params)

        # Defensive fallback: if stats weren't computed (e.g. Lightning skipped
        # on_fit_start somehow), compute them now.
        if self.train_mean is None or self.train_rmsd is None:
            self.train_mean, self.train_rmsd = self.compute_stats()

        # Surface compute_stats diagnostics to wandb (deferred from on_fit_start
        # because self.log is not allowed inside on_fit_start).
        if hasattr(self, "_stats_residual_mean"):
            self.log("residual_mean_after_refs", self._stats_residual_mean)
        if hasattr(self, "_stats_rms"):
            self.log("train_rms_eSEN", self._stats_rms)

    def on_train_epoch_start(self) -> None:
        for metric in self.train_metrics.values():
            metric.reset()
        self.set_optimizer_state("train")

    def on_validation_epoch_start(self) -> None:
        for metric in self.val_metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, "val")

    def on_test_epoch_start(self) -> None:
        for metric in self.test_metrics.values():
            metric.reset()
 
    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, "test")

    def on_validation_epoch_end(self):
        # Cast to float — for large models a full-epoch total_flops_used exceeds
        # int64 (2**63 ≈ 9.22e18) and wraps into negative values when Lightning
        # tensorises the scalar for sync_dist. float64 handles up to ~1e308.
        self.log("token_processed", float(self.token_processed), sync_dist=True, reduce_fx="sum")
        self.log("total_flops_used", float(self.total_flops_used), sync_dist=True, reduce_fx="sum")
        self.log("train_seconds", float(self.train_seconds), sync_dist=True, reduce_fx="mean")

    def set_optimizer_state(self, state: str):
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]

        for opt in opts:
            if isinstance(opt, schedulefree.AdamWScheduleFree):
                if state == "train":
                    opt.train()
                elif state == "eval":
                    opt.eval()
                else:
                    raise ValueError(f"Unknown train state {state}")

    @staticmethod
    def _random_rotation(include_reflection, device, dtype):
        """Sample a random SO(3) or O(3) matrix on the given device, no host syncs."""
        H = torch.randn(3, 3, device=device, dtype=dtype)
        Q, R = torch.linalg.qr(H)
        Q = Q * torch.sign(torch.diag(R))
        # Ensure det(Q) = +1 (proper rotation) without a Python branch
        Q[:, 0] = Q[:, 0] * torch.where(torch.det(Q) < 0, -1.0, 1.0).to(dtype)
        if include_reflection:
            # 50% chance of a single-column flip → det = -1
            Q[:, 0] = Q[:, 0] * torch.where(torch.rand((), device=device) < 0.5, -1.0, 1.0).to(dtype)
        return Q

    def _apply_augmentation(self, batch):
        """Apply random rotation (and optionally reflection) to positions and forces."""
        if not self.train_augmentation or not self.training:
            return batch
        include_reflection = getattr(self, '_augment_reflect', False)
        R = self._random_rotation(include_reflection, batch.pos.device, batch.pos.dtype)
        batch.pos = batch.pos @ R.T
        batch.forces = batch.forces @ R.T
        # cell/cell_offsets if present (periodic systems)
        if hasattr(batch, 'cell') and batch.cell is not None:
            batch.cell = batch.cell @ R.T
        return batch

    def training_step(self, batch, batch_idx):
        batch.z = batch.atomic_numbers.long()
        batch = self._apply_augmentation(batch)
        self._accumulate_tokens(batch)
        loss, log_dict = self._compute_loss(batch)
        loss_value = float(loss.detach().float().item())
        if (not np.isfinite(loss_value)) or (self.skip_loss_above > 0.0 and loss_value > self.skip_loss_above):
            threshold = self.skip_loss_above
            log.warning(
                "Skipping optimizer update at step=%s batch_idx=%s loss=%s threshold=%s",
                self.global_step,
                batch_idx,
                loss_value,
                threshold,
            )
            self.log("skip/loss", loss_value if np.isfinite(loss_value) else 0.0, on_step=True, on_epoch=False)
            self.log("skip/threshold", threshold, on_step=True, on_epoch=False)
            self.log("skip/update", 1.0, on_step=True, on_epoch=True)
            return None
        self._log_stage_metrics(log_dict, self.train_metrics, "train", batch.num_graphs, on_step=True, on_epoch=True)
        return loss

    def _shared_eval(self, batch, stage: str):
        batch.z = batch.atomic_numbers.long()
        loss, log_dict = self._compute_loss(batch)
        metrics = getattr(self, f"{stage}_metrics")
        self._log_stage_metrics(log_dict, metrics, stage, batch.num_graphs, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _compute_loss(self, batch):
        e_pred, e_gt, f_pred, f_gt = self(batch)

        if hasattr(batch, "energy_ref_corrected"):
            e_gt = batch.energy_ref_corrected.to(dtype=e_gt.dtype, device=e_gt.device).view_as(e_gt)
        else:
            e_gt = self.element_references.apply_refs(batch, e_gt)

        # eSEN-style normalization: no mean subtraction; divide by shared RMS-from-zero.
        e_loss = self.e_loss_fn(e_pred, e_gt / self.train_rmsd, batch.natoms)
        f_loss = self.f_loss_fn(f_pred, f_gt / self.train_rmsd, batch.natoms)

        # Ramp e_weight from 0 to target over e_weight_warmup_steps
        if self.e_weight_warmup_steps > 0 and self.global_step < self.e_weight_warmup_steps:
            e_w = self.e_weight * self.global_step / self.e_weight_warmup_steps
        else:
            e_w = self.e_weight
        loss = e_w * e_loss + self.f_weight * f_loss

        e_err = torch.nn.functional.l1_loss(e_pred * self.train_rmsd, e_gt).detach()*1000  # convert to meV (per molecule)
        e_err_per_atom = (((e_pred * self.train_rmsd - e_gt).abs() / batch.natoms).mean()).detach() * 1000  # meV/atom — leaderboard metric
        f_err = torch.nn.functional.l1_loss(f_pred * self.train_rmsd, f_gt).detach()*1000  # convert to meV/Å
        total = (e_err + f_err) / 2

        log_dict = dict(loss=loss, f_mae=f_err, e_mae=e_err, e_mae_per_atom=e_err_per_atom, total=total, f_loss=f_loss, e_loss=e_loss)
        return loss, log_dict

    def _log_stage_metrics(self, log_dict, metrics, stage: str, batch_size: int, on_step=False, on_epoch=True, prog_bar=False):
        for key, val in log_dict.items():
            metrics[key](val)
        self.log_dict(
            {f"{key}/{stage}": metrics[key] for key in log_dict},
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
        )

    def _accumulate_tokens(self, batch):
        self.token_processed += batch.atomic_numbers.shape[0]
        self.total_flops_used = self.flops_coef * self.token_processed * self.total_params
        # sync_dist + sum reduces rank-local counters into a global count under DDP;
        # no-op when world_size=1, so existing single-GPU runs stay comparable.
        self.log_dict(
            {
                "token_processed": float(self.token_processed),
                "total_flops_used": float(self.total_flops_used),
            },
            sync_dist=True, reduce_fx="sum",
        )
        self.log("train_seconds", float(self.train_seconds), sync_dist=True, reduce_fx="mean")

    def configure_optimizers(self):
        parameters_dict = build_optimizer_param_groups(
            self.net,
            self.mup_multipliers,
            self.hparams.optimizer.name == "adamw",
            {"lr": self.hparams.optimizer.lr, "weight_decay": self.hparams.optimizer.weight_decay},
        )
        self.free_scheduler = False

        if self.hparams.optimizer.name == "adam":
            optimizer = torch.optim.Adam(parameters_dict)
        elif self.hparams.optimizer.name == "adamw":
            optimizer = torch.optim.AdamW(parameters_dict)
        elif self.hparams.optimizer.name == "free":
            warmup = int(getattr(self.hparams.optimizer, "num_warmup_steps", 0))
            wlp = float(getattr(self.hparams.optimizer, "weight_lr_power", 2.0))
            r = float(getattr(self.hparams.optimizer, "r", 0.0))
            optimizer = schedulefree.AdamWScheduleFree(parameters_dict, warmup_steps=warmup, weight_lr_power=wlp, r=r)
            self.free_scheduler = True
            return optimizer
        else:
            raise NotImplemented

        total_steps = self.trainer.estimated_stepping_batches

        if self.hparams.optimizer.scheduler_name == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=self.hparams.optimizer.lr_min
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        elif self.hparams.optimizer.scheduler_name == "cosine_annealing_ws":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.optimizer.num_warmup_steps * total_steps),
                num_training_steps=total_steps,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            mode = self.hparams.compile_mode or "default"
            dyn = self.hparams.compile_dynamic
            self.net = torch.compile(self.net, mode=mode, dynamic=dyn)

    def train(self, mode: bool = True) -> None:
        self.net.train(mode)
        if self.free_scheduler:
            self.optimizers().train()

    def eval(self) -> None:
        self.net.eval()
        if self.free_scheduler:
            self.optimizers().eval()

    def on_validation_model_eval(self) -> None:
        # at the start of validation
        self.net.eval()
        if self.free_scheduler:
            self.optimizers().eval()

    def on_validation_model_train(self) -> None:
        # right after validation ends, before next training epoch
        self.net.train()
        if self.free_scheduler:
            self.optimizers().train()

    def on_test_model_eval(self) -> None:
        self.net.eval()
        if self.free_scheduler:
            self.optimizers().eval()

    def on_test_model_train(self) -> None:
        self.net.train()
        if self.free_scheduler:
            self.optimizers().train()

    def on_predict_model_eval(self) -> None:
        # redundant with on_predict_start(), but for completeness
        self.net.eval()
        if self.free_scheduler:
            self.optimizers().eval()
