import gc
import os
import sys
from typing import Union

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ml_collections
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.data import Data

import yaml

from platonic_transformers.datasets.omol import get_omol_loaders
from platonic_transformers.models.platoformer.force_field import PlatonicForceField
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.chg_spin_emb import ChgSpinEmbedding
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
)
from platonic_transformers.utils.utils import CosineWarmupScheduler, RandomSOd
from platonic_transformers.utils.callbacks import (
    MemoryMonitorCallback, TimerCallback, EMACallback,
)


def _cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                 num_cycles: float = 0.5, last_epoch: int = -1):
    """Linear warmup → cosine decay to 0. Matches the private's
    ``transformers.get_cosine_schedule_with_warmup`` (qcczbpfn used
    ``cosine_annealing_ws`` with this exact scheduler under the hood)."""
    import math as _math
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + _math.cos(_math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# Performance optimizations — exact qcczbpfn settings (see baseline-reference/
# wandb/files/config.yaml).
torch.set_float32_matmul_precision('high')   # qcczbpfn `matmul_precision: high`
torch.backends.cudnn.benchmark = False        # qcczbpfn `cudnn_benchmark: false`
torch.backends.cudnn.deterministic = True     # qcczbpfn `cudnn_deterministic: true`
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Dynamo knobs — qcczbpfn applies these whenever compile=True. Without them, the
# default cache_size_limit=8 trips on PlatonicLinear.forward under dynamic shapes
# (~11 recompiles seen on snellius), forcing a fall-back to eager mode. Same
# rationale as hipster's train_omol.py:130-141.
import torch._dynamo as _dynamo
_dynamo.config.cache_size_limit = 256
_dynamo.config.force_parameter_static_shapes = False
_dynamo.config.capture_scalar_outputs = True


class OMolModel(pl.LightningModule):
    """Lightning module for OMol energy and force prediction."""

    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()
        self.save_hyperparameters({'config': config.to_dict()})
        self.config = config

        # Random SO(3) augmentation; O(3) (rotation + reflection) is handled by
        # PlatonicForceField's `train_augmentation` setting in the private repo,
        # but here we apply rotation in `training_step` for parity with the
        # public augmentation pipeline.
        self.rotation_generator = RandomSOd(3)

        # --- Per-element reference energy subtraction (OMol25) ---
        # Loaded eagerly into a non-trainable buffer; used in `_compute_loss`
        # and the val/test metrics. Reads `omol_elem_refs` (NOT `oc20_*`).
        element_refs_path = getattr(self.config.dataset, "element_refs_path", None)
        if element_refs_path:
            if not os.path.isabs(element_refs_path):
                element_refs_path = os.path.join(REPO_ROOT, element_refs_path)
            with open(element_refs_path, "r") as f:
                refs_data = yaml.safe_load(f)
            refs = refs_data.get("omol_elem_refs", refs_data)
            self.register_buffer("element_refs", torch.tensor(refs, dtype=torch.float64))
        else:
            self.element_refs = None

        # --- PlatonicForceField (verbatim port of the private
        # `nets/platoformer/model.py::PlatonicForceField`) ---
        # All chgspin handling (RFF embedders + Linear+SiLU mix + input-add +
        # per-block FiLM), atom embedding (learned/khot), fp64 energy reduction,
        # and direct force output live inside the wrapper — same code as the
        # qcczbpfn run, just under a different package path.
        m = self.config.model
        self.net = PlatonicForceField(
            max_num_elements=int(getattr(m, "max_num_elements", 100)),
            hidden_dim=int(m.hidden_dim),
            nhead=int(m.num_heads),
            num_layers=int(m.num_layers),
            solid_name=str(m.solid_name),
            ffn_dim_factor=int(getattr(m, "ffn_dim_factor", 4)),
            dropout=float(m.dropout),
            rope_sigma=float(m.rope_sigma),
            learned_freqs=bool(m.learned_freqs),
            dense_mode=bool(m.dense_mode),
            embed_init_std=getattr(m, "embed_init_std", None),
            layer_scale_init_value=getattr(m, "layer_scale_init_value", None),
            attention=bool(m.attention),
            avg_num_nodes=float(getattr(m, "avg_num_nodes", 1.0)),
            rope_on_values=bool(getattr(m, "rope_on_values", False)),
            activation=str(getattr(m, "activation", "gelu")),
            readout_activation=getattr(m, "readout_activation", None),
            atom_embed=str(getattr(m, "atom_embed", "learned")),
            drop_path_rate=float(getattr(m, "drop_path_rate", 0.0)),
            freq_init=str(getattr(m, "freq_init", "random")),
            attention_backend=str(getattr(m, "attention_backend", "scatter")),
            qk_norm=bool(getattr(m, "qk_norm", False)),
            swiglu=bool(getattr(m, "swiglu", False)),
            qk_dim_factor=int(getattr(m, "qk_dim_factor", 1)),
            v_dim_factor=int(getattr(m, "v_dim_factor", 1)),
            rope_v_independent=bool(getattr(m, "rope_v_independent", False)),
            use_key=bool(getattr(m, "use_key", False)),
            norm_type=str(getattr(m, "norm_type", "layernorm")),
            chgspin_mode=str(getattr(m, "chgspin_mode", "off")),
            chgspin_emb_dim=getattr(m, "chgspin_embed_dim", None),
            chgspin_mix_init_std=float(getattr(m, "chgspin_mix_init_std", 0.02)),
            chgspin_layerwise=bool(getattr(m, "chgspin_layerwise", False)),
            chgspin_layerwise_gate=bool(getattr(m, "chgspin_layerwise_gate", False)),
            chgspin_film=bool(getattr(m, "chgspin_film", False)),
            interaction_radius=getattr(m, "interaction_radius", None),
            cutoff_p=int(getattr(m, "cutoff_p", 6)),
            max_num_neighbors=int(getattr(m, "max_num_neighbors", 1000)),
            local_global=bool(getattr(m, "local_global", False)),
        )

        # Normalization buffers — `scale = train_rmsd` (1.433569 for OMol25-4M
        # de-referenced targets). Without this normalization the loss is on
        # eV-scale residuals and the model fails to learn (see fi8zxww6).
        default_train_rmsd = 1.433569
        train_rmsd = float(getattr(self.config.dataset, "train_rmsd", default_train_rmsd))
        self.register_buffer('shift', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(train_rmsd, dtype=torch.float32))
        self.register_buffer('avg_num_nodes', torch.tensor(1.0, dtype=torch.float32))

        # Scaling-laws accumulators (matches qcczbpfn wandb keys
        # `token_processed`, `total_flops_used`, `train_seconds`, `total_parameters`).
        self.flops_coef = float(getattr(self.config.training, "flops_coef", 72.0))
        self.total_params = 0.0  # populated in on_fit_start
        self.register_buffer(
            "token_processed_buf", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer(
            "total_flops_used_buf", torch.tensor(0.0, dtype=torch.float64))
        self._fit_start_time = None

        # TorchMetric attributes — retained for compatibility with any helper
        # paths that reference them; the active val/test logging goes through
        # `_log_stage_metrics` directly.
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.train_metric_force = torchmetrics.MeanAbsoluteError()
        self.train_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric_force = torchmetrics.MeanAbsoluteError()
        self.valid_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()
        self.test_metrics_energy = torchmetrics.MeanAbsoluteError()
        self.test_metrics_force = torchmetrics.MeanAbsoluteError()
        self.test_metrics_energy_per_atom = torchmetrics.MeanAbsoluteError()

    def forward(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct energy + force prediction via `PlatonicForceField`.

        Builds the dict-style `data` that the private wrapper expects, defaulting
        `charge`/`spin` to zeros when the dataset doesn't supply them.
        """
        graph = graph.to(self.device)
        num_graphs = int(graph.batch.max().item()) + 1
        device = graph.pos.device
        charge = getattr(graph, "charge", None)
        if charge is None:
            charge = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        else:
            charge = charge.view(-1).float()
        spin = getattr(graph, "spin", None)
        if spin is None:
            spin = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        else:
            spin = spin.view(-1).float()
        data = {
            "atomic_numbers": graph.atomic_numbers.long(),
            "pos": graph.pos,
            "batch": graph.batch,
            "charge": charge,
            "spin": spin,
        }
        out = self.net(data)
        return out["energy"], out["forces"]

    def pred_energy_and_force(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct prediction — PlatonicForceField returns both energy and forces."""
        return self(graph)

    def _per_graph_ref_sum(self, graph: Data) -> torch.Tensor:
        """Sum of per-element reference energies over the atoms of each graph.

        Returns a tensor of shape (num_graphs,) in the same device as graph.pos.
        Returns zeros if element_refs is not configured or atomic_numbers is
        unavailable on the batch.
        """
        if self.element_refs is None or not hasattr(graph, "atomic_numbers"):
            num_graphs = int(graph.batch.max().item()) + 1
            return torch.zeros(num_graphs, device=graph.pos.device, dtype=torch.float32)
        z = graph.atomic_numbers.long().clamp_min(0)
        refs_per_atom = self.element_refs.to(graph.pos.device)[z]
        num_graphs = int(graph.batch.max().item()) + 1
        out = torch.zeros(num_graphs, dtype=self.element_refs.dtype, device=graph.pos.device)
        out.index_add_(0, graph.batch, refs_per_atom)
        return out.float()

    def _compute_loss(self, graph, pred_energy, pred_force):
        """Energy + force loss, faithfully ported from the private platonic-omol
        ``omol_module._compute_loss``:

        - Element references (omol_elem_refs) subtracted from the target energy.
        - Targets normalized by ``train_rmsd`` (self.scale).
        - Energy loss = per-atom MAE  : mean_g |e_pred_g/N_g - e_tgt_g/N_g|
          (fairchem ``PerAtomMAELoss`` + mean reduction)
        - Force loss  = L2-norm MAE   : mean_atoms ||f_pred_i - f_tgt_i||_2
          (fairchem ``L2NormLoss`` + mean reduction)
        - loss = e_weight * e_loss + f_weight * f_loss

        Returns (loss, e_loss, f_loss, e_mae_meV, f_mae_meV, e_mae_per_atom_meV, total).
        Metrics are reported in the *de-referenced* energy space (meV), matching
        the private run's wandb logging so the curves overlay directly.
        """
        natoms = graph.num_atoms.to(pred_energy.dtype)            # (num_graphs,)
        ref_sum = self._per_graph_ref_sum(graph)
        target_energy = graph.energy.view(-1) - ref_sum           # de-referenced GT, eV
        e_target_norm = target_energy / self.scale                # normalized
        f_target_norm = graph.forces / self.scale                 # normalized

        # per-atom MAE on energy (both pred and target divided by natoms)
        e_loss = ((pred_energy / natoms) - (e_target_norm / natoms)).abs().mean()
        # L2-norm MAE on forces
        f_loss = torch.linalg.vector_norm(pred_force - f_target_norm, ord=2, dim=-1).mean()

        e_weight = float(getattr(self.config.training, "lambda_E", 10.0))
        f_weight = float(getattr(self.config.training, "lambda_F", 20.0))
        loss = e_weight * e_loss + f_weight * f_loss

        # Metrics in meV, de-referenced space (e_pred * scale vs target_energy).
        e_pred_ev = pred_energy.detach() * self.scale
        f_pred_ev = pred_force.detach() * self.scale
        e_abs = (e_pred_ev - target_energy.detach()).abs()
        e_mae_mev = e_abs.mean() * 1000
        e_mae_per_atom_mev = (e_abs / natoms).mean() * 1000
        f_mae_mev = (f_pred_ev - graph.forces.detach()).abs().mean() * 1000
        total = (e_mae_mev + f_mae_mev) / 2
        return loss, e_loss, f_loss, e_mae_mev, f_mae_mev, e_mae_per_atom_mev, total

    def on_fit_start(self) -> None:
        import time
        self._fit_start_time = time.perf_counter()
        # Param count computed here (Lightning forbids self.log inside
        # on_fit_start; we log it from on_train_start below).
        self.total_params = float(sum(p.numel() for p in self.net.parameters()))

    def on_train_start(self) -> None:
        # `total_parameters` — one-time scalar matching the qcczbpfn wandb key.
        # Must be logged from on_train_start, not on_fit_start (Lightning
        # forbids self.log there).
        self.log("total_parameters", self.total_params)

    def setup(self, stage: str) -> None:
        """Apply ``torch.compile`` to the network at fit-start. Mirrors the
        private ``OMolModule.setup``: qcczbpfn was trained with
        ``compile=True, compile_mode='default'`` — large speedup on H100.

        Compile is opt-in via ``config.training.compile`` (default True for the
        qcczbpfn recipe). Set to false if you hit compile-related issues or
        need easier-to-read tracebacks.
        """
        if stage == "fit" and bool(getattr(self.config.training, "compile", False)):
            mode = str(getattr(self.config.training, "compile_mode", "default"))
            dynamic = getattr(self.config.training, "compile_dynamic", None)
            print(f"[OMolModel] torch.compile(self.net, mode='{mode}', dynamic={dynamic})")
            self.net = torch.compile(self.net, mode=mode, dynamic=dynamic)

    def _log_stage_metrics(self, stage, loss, e_loss, f_loss, e_mae, f_mae,
                           e_mae_per_atom, total, on_step, on_epoch, batch_size):
        """Emit the seven (loss, e_loss, f_loss, e_mae, f_mae, e_mae_per_atom,
        total) keys under ``{key}/{stage}`` — matches the private
        ``_log_stage_metrics`` and the qcczbpfn wandb metric set exactly.

        ``batch_size`` is passed explicitly because Lightning can't infer it
        from a PyG ``Batch`` object (the dict-like structure has multiple
        candidate sizes — pos: N atoms, batch: N atoms, energy: B graphs, ...).

        When ``on_step=True, on_epoch=True`` (train), Lightning emits both
        ``{key}/{stage}_step`` and ``{key}/{stage}_epoch`` — same as private.
        For val/test we pass ``on_step=False, on_epoch=True`` → keys land as
        ``{key}/{stage}`` (no suffix), again matching qcczbpfn.
        """
        # e_mae/f_mae go on the progress bar; the rest don't.
        self.log_dict({
            f"e_mae/{stage}": e_mae,
            f"f_mae/{stage}": f_mae,
        }, on_step=on_step, on_epoch=on_epoch, sync_dist=True, prog_bar=True,
           batch_size=batch_size)
        self.log_dict({
            f"loss/{stage}": loss,
            f"e_loss/{stage}": e_loss,
            f"f_loss/{stage}": f_loss,
            f"e_mae_per_atom/{stage}": e_mae_per_atom,
            f"total/{stage}": total,
        }, on_step=on_step, on_epoch=on_epoch, sync_dist=True, prog_bar=False,
           batch_size=batch_size)

    def training_step(self, graph: Data, batch_idx: int) -> torch.Tensor:
        if self.config.training.train_augm:
            batch_size = graph.batch.max().item() + 1
            rots = self.rotation_generator(n=batch_size).type_as(graph.pos)
            # qcczbpfn used `train_augmentation: o3` — half the samples in each
            # batch also get a reflection (det = -1). We implement this by
            # multiplying a random subset of rotation matrices by diag(-1,1,1):
            # an improper rotation = reflection × rotation, sampling uniformly
            # from O(3) rather than SO(3).
            if str(getattr(self.config.training, "train_augm_group", "o3")).lower() == "o3":
                flip_mask = (torch.rand(batch_size, device=rots.device) < 0.5)
                if flip_mask.any():
                    rots[flip_mask, 0, :] = -rots[flip_mask, 0, :]
            rot_per_sample = rots[graph.batch]
            graph.pos = torch.einsum('bij,bj->bi', rot_per_sample, graph.pos)
            graph.forces = torch.einsum('bij,bj->bi', rot_per_sample, graph.forces)

        pred_energy, pred_force = self.pred_energy_and_force(graph)

        loss, e_loss, f_loss, e_mae_mev, f_mae_mev, e_mae_per_atom_mev, total = \
            self._compute_loss(graph, pred_energy, pred_force)

        # Private-style logging: base key + on_step+on_epoch=True so Lightning
        # auto-emits both `<key>/train_step` and `<key>/train_epoch`. This
        # matches the qcczbpfn metric set 1:1.
        bsz = int(graph.batch.max().item()) + 1
        self._log_stage_metrics(
            "train", loss.detach(), e_loss.detach(), f_loss.detach(),
            e_mae_mev, f_mae_mev, e_mae_per_atom_mev, total,
            on_step=True, on_epoch=True, batch_size=bsz,
        )

        # Scaling-laws accumulators (qcczbpfn keys: token_processed,
        # total_flops_used, train_seconds). `flops_coef * tokens * params`
        # matches the private `_accumulate_tokens` exactly.
        n_tok = float(
            graph.atomic_numbers.shape[0]
            if hasattr(graph, "atomic_numbers") else graph.pos.shape[0]
        )
        self.token_processed_buf = self.token_processed_buf + n_tok
        self.total_flops_used_buf = (
            self.flops_coef * self.token_processed_buf * self.total_params
        )
        if self._fit_start_time is not None:
            import time
            train_seconds = float(time.perf_counter() - self._fit_start_time)
        else:
            train_seconds = 0.0
        self.log_dict({
            "token_processed": self.token_processed_buf,
            "total_flops_used": self.total_flops_used_buf,
        }, on_step=True, on_epoch=False, sync_dist=True, reduce_fx="sum")
        self.log("train_seconds", train_seconds,
                 on_step=True, on_epoch=False, sync_dist=True, reduce_fx="mean")

        if batch_idx % 250 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        # Re-log the scaling-laws counters at val time so wandb has a value
        # for `token_processed` / `total_flops_used` / `train_seconds` at the
        # same `_step` as `e_mae/val` etc. Without this, val curves grey out
        # when those axes are selected (val rows have no token_processed
        # value, so wandb can't pair them). Mirrors hipster's
        # `GraphModel.on_validation_epoch_end`.
        import time
        train_seconds = (
            float(time.perf_counter() - self._fit_start_time)
            if self._fit_start_time is not None else 0.0
        )
        self.log("token_processed", self.token_processed_buf,
                 sync_dist=True, reduce_fx="sum")
        self.log("total_flops_used", self.total_flops_used_buf,
                 sync_dist=True, reduce_fx="sum")
        self.log("train_seconds", train_seconds,
                 sync_dist=True, reduce_fx="mean")

    def validation_step(self, graph: Data, batch_idx: int) -> None:
        pred_energy, pred_force = self.pred_energy_and_force(graph)
        loss, e_loss, f_loss, e_mae_mev, f_mae_mev, e_mae_per_atom_mev, total = \
            self._compute_loss(graph, pred_energy, pred_force)
        bsz = int(graph.batch.max().item()) + 1
        self._log_stage_metrics(
            "val", loss.detach(), e_loss.detach(), f_loss.detach(),
            e_mae_mev, f_mae_mev, e_mae_per_atom_mev, total,
            on_step=False, on_epoch=True, batch_size=bsz,
        )

    def test_step(self, graph: Data, batch_idx: int) -> None:
        pred_energy, pred_force = self.pred_energy_and_force(graph)
        loss, e_loss, f_loss, e_mae_mev, f_mae_mev, e_mae_per_atom_mev, total = \
            self._compute_loss(graph, pred_energy, pred_force)
        bsz = int(graph.batch.max().item()) + 1
        self._log_stage_metrics(
            "test", loss.detach(), e_loss.detach(), f_loss.detach(),
            e_mae_mev, f_mae_mev, e_mae_per_atom_mev, total,
            on_step=False, on_epoch=True, batch_size=bsz,
        )
  
    def configure_optimizers(self) -> dict[str, object]:
        """Create optimizer and optional scheduler with custom decay groups."""

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, module in self.named_modules():
            for pn, param in module.named_parameters():
                full_name = f"{mn}.{pn}" if mn else pn

                if pn == 'freqs':
                    no_decay.add(full_name)
                elif pn.endswith('bias') or ('layer_scale' in pn):
                    no_decay.add(full_name)
                elif pn.endswith('weight') and isinstance(module, whitelist_weight_modules):
                    decay.add(full_name)
                elif pn.endswith('kernel'):
                    decay.add(full_name)
                elif pn.endswith('weight') and isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_name)

        param_dict = {pn: param for pn, param in self.named_parameters() if param.requires_grad}
        missing_params = param_dict.keys() - (decay | no_decay)
        if missing_params:
            print(f"Warning: Parameters {missing_params} were not explicitly assigned. Adding to no_decay.")
            no_decay.update(missing_params)

        assert len(decay & no_decay) == 0, f"Parameters in both decay and no_decay sets: {decay & no_decay}"
        
        optim_groups = [
            {
                "params": [param_dict[name] for name in sorted(decay) if name in param_dict],
                "weight_decay": self.config.optimizer.weight_decay,
            },
            {
                "params": [param_dict[name] for name in sorted(no_decay) if name in param_dict],
                "weight_decay": 0.0,
            },
        ]
        
        optim_groups = [group for group in optim_groups if group["params"]]

        optimizer_name = getattr(self.config.optimizer, "name", "adam").lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                optim_groups, lr=self.config.optimizer.lr,
                weight_decay=float(getattr(self.config.optimizer, "weight_decay", 0.0)),
            )
        else:
            optimizer = torch.optim.Adam(optim_groups, lr=self.config.optimizer.lr)

        scheduler_name = getattr(self.config.scheduler, "name", "cosine").lower()
        if scheduler_name == "cosine_annealing_ws":
            # Matches qcczbpfn's `cosine_annealing_ws`: linear warmup over
            # `num_warmup_steps` (fraction of total steps if <1, else integer)
            # → cosine decay to 0. `interval=step` updates the LR each
            # optimizer step.
            #
            # Compute total_steps explicitly: Lightning 2.6's
            # `trainer.estimated_stepping_batches` can under-count under DDP +
            # custom batch_sampler + use_distributed_sampler=False (it returned
            # ~1 epoch's worth of batches on the 4× H100 snellius run
            # `iftu53zm`, vs ~20 epochs on the 1× H100 run `y0b70zqc` with the
            # same code). The bug makes the cosine decay finish in 1 epoch on
            # multi-GPU; LR ≈ 0 from epoch 2 onward and the model undertrains.
            # Hipster's Lightning 2.5.5 didn't have this regression, which is
            # why qcczbpfn's 4-GPU schedule was correct. Fall back to
            # estimated_stepping_batches if the dataloader length isn't
            # available at configure_optimizers time.
            try:
                batches_per_epoch = len(self.trainer.train_dataloader)
                max_epochs = int(self.trainer.max_epochs or 1)
                accum = max(1, int(self.trainer.accumulate_grad_batches or 1))
                total_steps = (batches_per_epoch * max_epochs) // accum
                print(
                    f"[OMolModel] cosine schedule total_steps={total_steps} "
                    f"(batches_per_epoch={batches_per_epoch} × max_epochs={max_epochs} "
                    f"÷ accum={accum}; estimated_stepping_batches reported "
                    f"{int(self.trainer.estimated_stepping_batches)})"
                )
            except (AttributeError, TypeError, ValueError, NotImplementedError) as exc:
                total_steps = int(self.trainer.estimated_stepping_batches)
                print(
                    f"[OMolModel] cosine schedule total_steps={total_steps} "
                    f"(falling back to estimated_stepping_batches; could not "
                    f"derive from dataloader: {exc!r})"
                )
            num_warmup = float(getattr(self.config.scheduler, "num_warmup_steps", 0.01))
            if num_warmup <= 1.0:
                num_warmup_steps = int(num_warmup * total_steps)
            else:
                num_warmup_steps = int(num_warmup)
            scheduler = _cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        elif getattr(self.config.scheduler, "use_cosine", False):
            scheduler = CosineWarmupScheduler(
                optimizer, self.config.scheduler.warmup_epochs, self.trainer.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "e_mae/val"}
        return {"optimizer": optimizer, "monitor": "e_mae/val"}


class ESENModel(OMolModel):
    """Lightning module for OMol energy + force prediction with eSEN baseline.

    Wraps ``EquivariantNet`` (eSCNMDBackbone + MLP_EFS_Head, conservative-force
    recipe) instead of ``PlatonicTransformer``. Inherits training_step / val /
    test / configure_optimizers from OMolModel so the loss + metric pipeline
    is identical — only the model construction and forward pass differ.

    Requires ``fairchem-core>=2.19``. See ``configs/omol_esen.yaml`` for the
    eSEN-small recipe.
    """

    def __init__(self, config: ml_collections.ConfigDict) -> None:
        # Skip OMolModel.__init__ (would build PlatonicTransformer) — call
        # the LightningModule base init directly.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters({'config': config.to_dict()})
        self.config = config

        self.rotation_generator = RandomSOd(3)

        from platonic_transformers.models.baseline.esen.eqv_net import EquivariantNet
        m = config.model
        self.net = EquivariantNet(
            max_num_elements=getattr(m, "max_num_elements", 100),
            sphere_channels=m.sphere_channels,
            hidden_channels=m.hidden_channels,
            lmax=m.lmax,
            mmax=m.mmax,
            num_layers=m.num_layers,
            cutoff=m.cutoff,
            max_neighbors=m.max_neighbors,
            otf_graph=m.otf_graph,
            direct_forces=m.direct_forces,
            regress_forces=m.regress_forces,
            regress_stress=m.regress_stress,
            activation_checkpointing=m.activation_checkpointing,
            norm_type=m.norm_type,
            act_type=m.act_type,
            ff_type=m.ff_type,
            chg_spin_emb_type=m.chg_spin_emb_type,
        )

        # Common buffers + metrics. `scale` initialized to the private's
        # train_rmsd value (1.433569) for correct loss magnitude — same
        # rationale as OMolModel above.
        default_train_rmsd = 1.433569
        train_rmsd = float(getattr(self.config.dataset, "train_rmsd", default_train_rmsd))
        self.register_buffer('shift', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(train_rmsd, dtype=torch.float32))
        self.register_buffer('avg_num_nodes', torch.tensor(1.0, dtype=torch.float32))

        # chgspin is handled internally by eSCNMDBackbone (via chg_spin_emb_type),
        # not through the Platonic FiLM path. Tell the inherited helpers we don't
        # have a top-level mix.
        self.charge_embedder = None
        self.spin_embedder = None
        self.chgspin_mix = None
        self.chgspin_mode = "off"
        self.chgspin_film_enabled = False

        # Scaling-laws accumulators — same buffers as OMolModel so the inherited
        # training_step's `token_processed`/`total_flops_used`/`train_seconds`
        # logging works under eSEN too.
        self.flops_coef = float(getattr(self.config.training, "flops_coef", 72.0))
        self.total_params = 0.0  # populated in on_fit_start (inherited)
        self.register_buffer(
            "token_processed_buf", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer(
            "total_flops_used_buf", torch.tensor(0.0, dtype=torch.float64))
        self._fit_start_time = None

        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.train_metric_force = torchmetrics.MeanAbsoluteError()
        self.train_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric_force = torchmetrics.MeanAbsoluteError()
        self.valid_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()
        self.test_metrics_energy = torchmetrics.MeanAbsoluteError()
        self.test_metrics_force = torchmetrics.MeanAbsoluteError()
        self.test_metrics_energy_per_atom = torchmetrics.MeanAbsoluteError()

        # Element reference energies — same path as OMolModel.
        element_refs_path = getattr(self.config.dataset, "element_refs_path", None)
        if element_refs_path:
            if not os.path.isabs(element_refs_path):
                element_refs_path = os.path.join(REPO_ROOT, element_refs_path)
            with open(element_refs_path, "r") as f:
                refs_data = yaml.safe_load(f)
            # OMol25 uses the `omol_elem_refs` key — full per-atom energy
            # references (~1000s eV/atom). NOT `oc20_elem_refs` (tiny
            # adsorption-energy refs, wrong scale for OMol25 total energies).
            # This matches src/utils/file_utils.load_reference in the private
            # platonic-omol repo.
            refs = refs_data.get("omol_elem_refs", refs_data)
            self.register_buffer("element_refs", torch.tensor(refs, dtype=torch.float64))
        else:
            self.element_refs = None

    def forward(self, graph):
        graph = graph.to(self.device)
        num_graphs = int(graph.batch.max().item()) + 1
        device = graph.pos.device
        # eSCNMDBackbone expects an object that supports BOTH attribute access
        # (canonical_pbc uses hasattr(data, "pbc")) AND .get() (line 701 of
        # escn_md.py does data_dict.get("dataset", None)). Fairchem's own
        # AtomicData class has both; PyG's plain Batch only has attribute
        # access. Augment the PyG batch with OMol25 fields + a `.get()` method.
        if not hasattr(graph, "charge"):
            graph.charge = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        if not hasattr(graph, "spin"):
            graph.spin = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        if not hasattr(graph, "natoms"):
            graph.natoms = getattr(graph, "num_atoms", None)
        if not hasattr(graph, "pbc"):
            graph.pbc = torch.zeros(num_graphs, 3, dtype=torch.bool, device=device)
        if not hasattr(graph, "cell"):
            graph.cell = torch.zeros(num_graphs, 3, 3, dtype=torch.float32, device=device)
        # Bind a `.get` method for fairchem's dict-style access calls.
        # (PyG Data only has __getitem__; AtomicData has both __getitem__ and .get.)
        if not hasattr(graph, "get"):
            import types
            def _pyg_get(self, key, default=None):
                # fairchem occasionally calls .get() with non-string keys;
                # getattr requires str, so guard for it.
                if not isinstance(key, str):
                    return default
                return getattr(self, key, default)
            graph.get = types.MethodType(_pyg_get, graph)
        out = self.net(graph)
        return out["energy"], out["forces"]

    def pred_energy_and_force(self, graph):
        # eSEN backbone+head computes both in one forward (forces via autograd
        # through MLP_EFS_Head). No separate autograd step needed.
        return self(graph)


def main(config: ml_collections.ConfigDict) -> None:
    """Train and evaluate the Platonic Transformer on the OMol dataset."""
    pl.seed_everything(config.seed)

    train_loader, val_loader, test_loader, _, _ = get_omol_loaders(
        root=config.dataset.data_dir,
        batch_size=config.training.batch_size,
        num_workers=2 if config.system.gpus > 1 else config.system.num_workers,
        use_charges=False,
        seed=config.seed,
        debug_subset=config.dataset.debug_subset,
        referencing=config.dataset.referencing,
        include_hof=config.dataset.include_hof,
        scale_shift=config.dataset.scale_shift,
        recalculate=config.dataset.recalculate_stats,
        use_k_hot=config.dataset.use_khot_encoding,
        # qcczbpfn used dynamic batching with max_atoms_per_batch=3000 +
        # max_edges_per_batch=600000 — packs each batch up to the atom budget
        # rather than a fixed graph count. Falls back to fixed batching when
        # disabled.
        dynamic_batching=bool(getattr(config.training, "dynamic_batching", False)),
        max_atoms_per_batch=getattr(config.training, "max_atoms_per_batch", None),
        max_atoms_per_batch_val=getattr(config.training, "max_atoms_per_batch_val", None),
        max_edges_per_batch=getattr(config.training, "max_edges_per_batch", None),
        max_edges_per_batch_val=getattr(config.training, "max_edges_per_batch_val", None),
        # qcczbpfn layout: `<data_dir>/train_4M/` + `<data_dir>/val/`. Override
        # via dataset.{train_subdir,val_subdir} if your data lives elsewhere.
        train_subdir=str(getattr(config.dataset, "train_subdir", "train_4M")),
        val_subdir=str(getattr(config.dataset, "val_subdir", "val")),
    )

    accelerator = "gpu" if config.system.gpus > 0 and torch.cuda.is_available() else "cpu"
    devices = config.system.gpus if accelerator == "gpu" else "auto"
        
    if config.logging.enabled:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(
            entity=config.logging.wandb_identity,                # null → wandb default user
            project=config.logging.project_name,                 # honors config (was hardcoded)
            name=config.logging.wandb_run_name,                  # null → wandb auto-generates
            config=config.to_dict(),
            save_dir=save_dir,
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='e_mae/val', mode='min', filename='best-energy-{epoch:02d}'),
        pl.callbacks.ModelCheckpoint(monitor='f_mae/val', mode='min', filename='best-force-{epoch:02d}'),
        pl.callbacks.ModelCheckpoint(monitor='e_mae_per_atom/val', mode='min', filename='best-energy-per-atom-{epoch:02d}'),
        pl.callbacks.ModelCheckpoint(save_last=True, filename='last'),
        TimerCallback(),
        MemoryMonitorCallback(log_frequency=50)
    ]
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
    if config.system.timer:
        callbacks.append(Timer(duration=config.system.timer))
    # EMA matches qcczbpfn's `ema/decay=0.99, ema/warmup_steps=2000`. The
    # callback swaps EMA weights into the model at val/test start, swaps back
    # at end — same shape as the private. Enable via config.optimizer.ema.
    if bool(getattr(config.optimizer, "ema", False)):
        callbacks.append(EMACallback(
            decay=float(getattr(config.optimizer, "ema_decay", 0.99)),
            warmup_steps=int(getattr(config.optimizer, "ema_warmup_steps", 2000)),
        ))
   
    # Select model class by config.model.name. "platoformer" (default) uses
    # OMolModel; "esen" uses the eSCNMDBackbone-based baseline.
    model_name = getattr(config.model, "name", "platoformer").lower()
    if model_name == "esen":
        ModelCls = ESENModel
    elif model_name == "platoformer":
        ModelCls = OMolModel
    else:
        raise ValueError(
            f"Unknown model.name='{model_name}'. Supported: 'platoformer', 'esen'."
        )

    if config.testing.load_weights:
        model = ModelCls.load_from_checkpoint(checkpoint_path=config.testing.load_weights, config=config)
    else:
        model = ModelCls(config)

    # Only override the model's train_rmsd-based scale/shift if dataset-level
    # scale_shift normalization was actually requested. Otherwise the model
    # keeps the canonical OMol25-4M train_rmsd=1.433569 (qcczbpfn behavior);
    # the dataset's default scale=1.0 would silently nuke that normalization.
    if getattr(config.dataset, "scale_shift", False) and hasattr(train_loader.dataset, 'scale'):
        model.scale = torch.tensor(train_loader.dataset.scale).to(model.device)
        model.shift = torch.tensor(train_loader.dataset.shift).to(model.device)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        precision=config.system.precision,
        inference_mode=False,
        # Validate every N optimizer steps + cap val to a fixed batch budget,
        # matching qcczbpfn's `trainer/val_check_interval=5000` and
        # `trainer/limit_val_batches=500`. Both configurable via yaml.
        val_check_interval=getattr(config.training, "val_check_interval", 5000),
        limit_val_batches=getattr(config.training, "limit_val_batches", 500),
        # Gradient accumulation lets a single-GPU run reproduce qcczbpfn's
        # multi-GPU effective batch size (4 GPUs × max_atoms=3000 = 12k atoms
        # per global step → 1 GPU × 3000 × accumulate=4).
        accumulate_grad_batches=int(getattr(
            config.system, "accumulate_grad_batches", 1)),
        strategy=DDPStrategy(find_unused_parameters=True) if config.system.gpus > 1 else 'auto',
        # Our DynamicAtomBatchSamplerForAseDB is already DDP-aware (shards by
        # rank internally). Lightning would otherwise auto-wrap the loader's
        # sampler with DistributedSampler, which fails for our custom sampler
        # (TypeError: doesn't subclass BatchSampler).
        use_distributed_sampler=False,
    )

    if not config.testing.test_ckpt:
        trainer.fit(model, train_loader, val_loader, ckpt_path=config.testing.resume_ckpt)
        best_ckpt_path = callbacks[2].best_model_path or "last"
        trainer.test(model, test_loader, ckpt_path=best_ckpt_path)
    else:
        model = OMolModel.load_from_checkpoint(
            config.testing.test_ckpt,
            hparams_file=os.path.join(os.path.dirname(config.testing.test_ckpt), "hparams.yaml"),
            config=config,
        )
        trainer.test(model, test_loader)

if __name__ == "__main__":
    # Parse command-line arguments (allow unknown for simple overrides)
    parser = get_arg_parser(default_config_path="configs/omol.yaml")
    args, unknown_args = parser.parse_known_args()
    
    # Load configuration and parse CLI overrides automatically
    config = load_with_defaults(
        dataset_config=args.config,
        cli_args=unknown_args  # Automatically infers parameter locations
    )
    
    # Run training
    main(config)
