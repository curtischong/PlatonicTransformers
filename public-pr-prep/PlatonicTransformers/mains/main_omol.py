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
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.chg_spin_emb import ChgSpinEmbedding
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
)
from platonic_transformers.utils.utils import CosineWarmupScheduler, RandomSOd
from platonic_transformers.utils.callbacks import MemoryMonitorCallback, TimerCallback

# Performance optimizations
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


class OMolModel(pl.LightningModule):
    """Lightning module for OMol energy and force prediction."""

    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()
        self.save_hyperparameters({'config': config.to_dict()})
        self.config = config

        # Setup rotation augmentation
        self.rotation_generator = RandomSOd(3)
        
        # Calculate total input channels
        in_channels_scalar = (
            92  # base atom features onehot
            + 3 * ("coords" in self.config.dataset.scalar_features)  # x,y,z coordinates as scalars
            + 1 * ("charges" in self.config.dataset.scalar_features)  # charges as scalars
        )
        in_channels_vector = 0  # No vector features used in this setup

        # --- OMol25 charge/spin embeddings (random Fourier features + Linear+SiLU mix) ---
        # When config.model.chgspin_mode != "off", build per-atom charge AND spin
        # RFF embeddings (ChgSpinEmbedding, pos_emb variant), concat them, and
        # mix through a single Linear + SiLU to produce a per-node conditioning
        # signal `chgspin_mixed`. This signal is then used by per-block FiLM
        # (when config.model.chgspin_film=True) inside PlatonicTransformer.
        chgspin_mode = getattr(self.config.model, "chgspin_mode", "off")
        chgspin_film = bool(getattr(self.config.model, "chgspin_film", False))
        # Per-G-slot embedding dim — defaults to hidden_dim // |G| if unset.
        num_G = PLATONIC_GROUPS[self.config.model.solid_name.lower()].G
        chgspin_embed_dim = getattr(self.config.model, "chgspin_embed_dim", None)
        if chgspin_embed_dim is None:
            chgspin_embed_dim = self.config.model.hidden_dim // num_G
        self.chgspin_mode = chgspin_mode
        self.chgspin_film_enabled = chgspin_film
        self.chgspin_embed_dim = chgspin_embed_dim
        needs_chgspin = (chgspin_mode != "off") or chgspin_film
        if needs_chgspin:
            self.charge_embedder = ChgSpinEmbedding(
                embedding_type="pos_emb", embedding_target="charge",
                embedding_size=chgspin_embed_dim,
            )
            self.spin_embedder = ChgSpinEmbedding(
                embedding_type="pos_emb", embedding_target="spin",
                embedding_size=chgspin_embed_dim,
            )
            mix_init_std = float(getattr(
                self.config.model, "chgspin_mix_init_std", 0.02))
            self.chgspin_mix = torch.nn.Linear(2 * chgspin_embed_dim, chgspin_embed_dim)
            torch.nn.init.normal_(self.chgspin_mix.weight, std=mix_init_std)
            torch.nn.init.constant_(self.chgspin_mix.bias, 0.0)
        else:
            self.charge_embedder = None
            self.spin_embedder = None
            self.chgspin_mix = None

        # --- Per-element reference energy subtraction (OC20 OMol25 refs) ---
        # When config.dataset.referencing=True and a path is provided, load the
        # per-Z reference energies and register as a non-trainable buffer.
        # In training/val/test the per-graph target energy has ∑_i refs[Z_i]
        # subtracted before computing the loss, then added back for reporting.
        element_refs_path = getattr(self.config.dataset, "element_refs_path", None)
        if element_refs_path:
            if not os.path.isabs(element_refs_path):
                element_refs_path = os.path.join(REPO_ROOT, element_refs_path)
            with open(element_refs_path, "r") as f:
                refs_data = yaml.safe_load(f)
            refs = refs_data.get("oc20_elem_refs", refs_data)
            refs_tensor = torch.tensor(refs, dtype=torch.float64)
            self.register_buffer("element_refs", refs_tensor)
        else:
            self.element_refs = None

        # --- Dynamically configure model outputs based on force prediction mode ---
        if self.config.model.predict_forces:
            # Direct force prediction: 1 scalar (energy) and 1 vector (force)
            out_channels_scalar = 1
            out_channels_vec = 1
            scalar_task_level = "graph"
            vector_task_level = "node"
        else:
            # Energy prediction only (forces from gradient)
            out_channels_scalar = 1
            out_channels_vec = 0
            scalar_task_level = "graph"
            vector_task_level = "graph"  # Not used for output, but required
        # --- End of dynamic configuration ---

        # Model specification
        solid_name = self.config.model.solid_name.lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(f"Unsupported solid_name '{solid_name}'. Supported: {list(PLATONIC_GROUPS.keys())}")

        self.net = PlatonicTransformer(
            input_dim=in_channels_scalar,
            input_dim_vec=in_channels_vector,
            hidden_dim=self.config.model.hidden_dim,
            output_dim=out_channels_scalar,
            output_dim_vec=out_channels_vec,
            nhead=self.config.model.num_heads,
            num_layers=self.config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=3,
            dense_mode=self.config.model.dense_mode,
            scalar_task_level=scalar_task_level,
            vector_task_level=vector_task_level,
            ffn_readout=self.config.model.ffn_readout,
            mean_aggregation=self.config.model.mean_aggregation,
            dropout=self.config.model.dropout,
            drop_path_rate=self.config.model.drop_path_rate,
            layer_scale_init_value=self.config.model.layer_scale_init_value,
            attention=self.config.model.attention,
            ffn_dim_factor=getattr(self.config.model, "ffn_dim_factor", 4),
            rope_sigma=self.config.model.rope_sigma,
            ape_sigma=self.config.model.ape_sigma,
            learned_freqs=self.config.model.learned_freqs,
            freq_init=self.config.model.freq_init,
            use_key=self.config.model.use_key,
            # OMol25 additions (defaults preserve old behavior):
            attention_backend=getattr(self.config.model, "attention_backend", "scatter"),
            qk_norm=bool(getattr(self.config.model, "qk_norm", False)),
            swiglu=bool(getattr(self.config.model, "swiglu", False)),
            rope_on_values=bool(getattr(self.config.model, "rope_on_values", False)),
            activation=getattr(self.config.model, "activation", "gelu"),
            chgspin_film=chgspin_film,
            chgspin_embed_dim=chgspin_embed_dim if chgspin_film else None,
        )

        # Initialize normalization parameters
        self.register_buffer('shift', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('avg_num_nodes', torch.tensor(1.0, dtype=torch.float32))
        
        # Setup metrics
        self.train_metric = torchmetrics.MeanAbsoluteError()
        self.train_metric_force = torchmetrics.MeanAbsoluteError()
        self.train_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()
        
        self.valid_metric = torchmetrics.MeanAbsoluteError()
        self.valid_metric_force = torchmetrics.MeanAbsoluteError()
        self.valid_metric_energy_per_atom = torchmetrics.MeanAbsoluteError()
        
        self.test_metrics_energy = torchmetrics.MeanAbsoluteError()
        self.test_metrics_force = torchmetrics.MeanAbsoluteError()
        self.test_metrics_energy_per_atom = torchmetrics.MeanAbsoluteError()

    def _chgspin_mixed_per_node(self, graph: Data) -> Union[torch.Tensor, None]:
        """Build per-node chg/spin conditioning signal.

        Reads per-graph ``graph.charge`` and ``graph.spin`` (floats), passes each
        through its own RFF embedder, concatenates, mixes through one Linear+SiLU,
        and broadcasts to per-node via ``graph.batch``. Returns ``None`` if charge
        and spin embedders are not configured (chgspin_mode='off' and no FiLM).
        """
        if self.charge_embedder is None:
            return None
        # Per-graph charge/spin (float). Tolerate dataset variants that omit spin.
        charge = getattr(graph, "charge", None)
        spin = getattr(graph, "spin", None)
        if charge is None:
            num_graphs = int(graph.batch.max().item()) + 1
            charge = torch.zeros(num_graphs, dtype=torch.float32, device=graph.pos.device)
        else:
            charge = charge.view(-1).float()
        if spin is None:
            spin = torch.zeros_like(charge)
        else:
            spin = spin.view(-1).float()
        chg_emb = self.charge_embedder(charge)           # (B, embed_dim)
        spn_emb = self.spin_embedder(spin)               # (B, embed_dim)
        mixed = torch.nn.functional.silu(
            self.chgspin_mix(torch.cat([chg_emb, spn_emb], dim=-1))
        )                                                # (B, embed_dim)
        # Broadcast per-graph signal to per-node via batch index.
        return mixed[graph.batch]                        # (N, embed_dim)

    def forward(self, graph: Data) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        graph = graph.to(self.device)
        # Prepare input features
        x = [graph.x]
        if "coords" in self.config.dataset.scalar_features:
            x.append(graph.pos)
        if "charges" in self.config.dataset.scalar_features:
            x.append(graph.charges[:, None])

        x = torch.cat(x, dim=-1)

        # Build per-node chg/spin conditioning signal (None if disabled).
        chgspin_mixed = self._chgspin_mixed_per_node(graph)

        # Forward pass
        pred_scalar, pred_vec = self.net(
            x, graph.pos, graph.batch, vec=None,
            avg_num_nodes=self.avg_num_nodes.to(graph.pos.device),
            chgspin_mixed_per_node=chgspin_mixed,
        )
        
        pred_energy = pred_scalar.view(-1)
        
        if self.config.model.predict_forces:
            # Squeeze the middle dimension: [N, 1, 3] -> [N, 3]
            pred_force = pred_vec.squeeze(1)
            return pred_energy, pred_force
        return pred_energy

    def pred_energy_and_force(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted energy and forces, using autograd if needed."""
        if self.config.model.predict_forces:
            # Model directly outputs energy and forces
            pred_energy, pred_force = self(graph)
            return pred_energy, pred_force

        # Calculate forces from energy gradient (autograd)
        with torch.enable_grad():
            graph.pos = graph.pos.clone().requires_grad_(True)
            pred_energy = self(graph)
            sign = -1.0
            pred_force = sign * torch.autograd.grad(
                pred_energy,
                graph.pos,
                grad_outputs=torch.ones_like(pred_energy),
                create_graph=self.training,
                retain_graph=self.training,
            )[0]

        if not self.training:
            pred_energy = pred_energy.detach()
            pred_force = pred_force.detach()

        return pred_energy, pred_force

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

    def training_step(self, graph: Data, batch_idx: int) -> torch.Tensor:
        if self.config.training.train_augm:
            batch_size = graph.batch.max().item() + 1
            rots = self.rotation_generator(n=batch_size).type_as(graph.pos)
            rot_per_sample = rots[graph.batch]
            graph.pos = torch.einsum('bij,bj->bi', rot_per_sample, graph.pos)
            graph.forces = torch.einsum('bij,bj->bi', rot_per_sample, graph.forces)
        
        pred_energy, pred_force = self.pred_energy_and_force(graph)
        
        # Loss calculation
        # Element-reference subtraction: when loaded, regress against the
        # graph energy *with per-element references removed* (a flat, less
        # extreme target distribution). For metric reporting we add the refs
        # back so absolute meV values are comparable across runs.
        ref_sum = self._per_graph_ref_sum(graph)
        target_energy = graph.energy - ref_sum
        energy_loss = torch.mean((pred_energy - ((target_energy - self.shift) / self.scale))**2)
        force_loss = torch.mean(torch.sqrt(torch.sum((pred_force - graph.forces / self.scale)**2, -1)))
        loss = energy_loss + self.config.training.lambda_F * force_loss

        # Logging metrics (converted to meV and meV/Å). Add element_refs back
        # for absolute energy reporting (the regression target had them removed).
        pred_energy_mev = (pred_energy.detach() * self.scale + self.shift + ref_sum) * 1000
        true_energy_mev = graph.energy * 1000
        pred_force_mev_ang = pred_force.detach() * self.scale * 1000
        true_force_mev_ang = graph.forces * 1000

        pred_energy_per_atom_mev = pred_energy_mev / graph.num_atoms
        true_energy_per_atom_mev = true_energy_mev / graph.num_atoms

        self.train_metric(pred_energy_mev, true_energy_mev)
        self.train_metric_force(pred_force_mev_ang, true_force_mev_ang)
        self.train_metric_energy_per_atom(pred_energy_per_atom_mev, true_energy_per_atom_mev)
        
        self.log("train MAE (energy) [meV]", self.train_metric, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train MAE (force) [meV/Å]", self.train_metric_force, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train MAE (energy/atom) [meV]", self.train_metric_energy_per_atom, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)

        if batch_idx % 250 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
        return loss

    def on_train_epoch_end(self) -> None:
        pass
    
    def validation_step(self, graph: Data, batch_idx: int) -> None:
        pred_energy, pred_force = self.pred_energy_and_force(graph)

        ref_sum = self._per_graph_ref_sum(graph)
        pred_energy_mev = (pred_energy * self.scale + self.shift + ref_sum) * 1000
        true_energy_mev = graph.energy * 1000
        pred_force_mev_ang = pred_force * self.scale * 1000
        true_force_mev_ang = graph.forces * 1000

        pred_energy_per_atom_mev = pred_energy_mev / graph.num_atoms
        true_energy_per_atom_mev = true_energy_mev / graph.num_atoms

        self.valid_metric(pred_energy_mev, true_energy_mev)
        self.valid_metric_force(pred_force_mev_ang, true_force_mev_ang)
        self.valid_metric_energy_per_atom(pred_energy_per_atom_mev, true_energy_per_atom_mev)

    def on_validation_epoch_end(self) -> None:
        self.log("valid MAE (energy) [meV]", self.valid_metric, prog_bar=True, sync_dist=True)
        self.log("valid MAE (force) [meV/Å]", self.valid_metric_force, prog_bar=True, sync_dist=True)
        self.log("valid MAE (energy/atom) [meV]", self.valid_metric_energy_per_atom, prog_bar=True, sync_dist=True)
    
    def test_step(self, graph: Data, batch_idx: int) -> None:
        pred_energy, pred_force = self.pred_energy_and_force(graph)

        ref_sum = self._per_graph_ref_sum(graph)
        pred_energy_mev = (pred_energy * self.scale + self.shift + ref_sum) * 1000
        true_energy_mev = graph.energy * 1000
        pred_force_mev_ang = pred_force * self.scale * 1000
        true_force_mev_ang = graph.forces * 1000
        
        pred_energy_per_atom_mev = pred_energy_mev / graph.num_atoms
        true_energy_per_atom_mev = true_energy_mev / graph.num_atoms
        
        self.test_metrics_energy(pred_energy_mev, true_energy_mev)
        self.test_metrics_force(pred_force_mev_ang, true_force_mev_ang)
        self.test_metrics_energy_per_atom(pred_energy_per_atom_mev, true_energy_per_atom_mev)

    def on_test_epoch_end(self) -> None:
        self.log("test MAE (energy) [meV]", self.test_metrics_energy, prog_bar=True, sync_dist=True)
        self.log("test MAE (force) [meV/Å]", self.test_metrics_force, prog_bar=True, sync_dist=True)
        self.log("test MAE (energy/atom) [meV]", self.test_metrics_energy_per_atom, prog_bar=True, sync_dist=True)
  
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

        optimizer = torch.optim.Adam(optim_groups, lr=self.config.optimizer.lr)
        if self.config.scheduler.use_cosine:
            scheduler = CosineWarmupScheduler(optimizer, self.config.scheduler.warmup_epochs, self.trainer.max_epochs)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid MAE (energy) [meV]"}
        else:
            return {"optimizer": optimizer, "monitor": "valid MAE (energy) [meV]"}


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
    )

    accelerator = "gpu" if config.system.gpus > 0 and torch.cuda.is_available() else "cpu"
    devices = config.system.gpus if accelerator == "gpu" else "auto"
        
    if config.logging.enabled:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        logger = pl.loggers.WandbLogger(project="Platonic-omol", config=config.to_dict(), save_dir=save_dir)
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='valid MAE (energy) [meV]', mode='min', filename='best-energy-{epoch:02d}'),
        pl.callbacks.ModelCheckpoint(monitor='valid MAE (force) [meV/Å]', mode='min', filename='best-force-{epoch:02d}'),
        pl.callbacks.ModelCheckpoint(monitor='valid MAE (energy/atom) [meV]', mode='min', filename='best-energy-per-atom-{epoch:02d}'),
        pl.callbacks.ModelCheckpoint(save_last=True, filename='last'),
        TimerCallback(),
        MemoryMonitorCallback(log_frequency=50)
    ]
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    if config.system.timer:
        callbacks.append(Timer(duration=config.system.timer))
   
    if config.testing.load_weights:
        model = OMolModel.load_from_checkpoint(checkpoint_path=config.testing.load_weights, config=config)
    else:
        model = OMolModel(config)

    if hasattr(train_loader.dataset, 'scale'):
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
        strategy=DDPStrategy(find_unused_parameters=True) if config.system.gpus > 1 else 'auto'
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
