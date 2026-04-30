#!/usr/bin/env python
"""Train paired QM9 models with Cartesian vs fractional+lattice coordinates.

This is a regression test for the lattice-aware ``PlatonicTransformer`` API.
Both models see the same QM9 batches, have identical initial weights, and use a
fixed non-orthogonal lattice. One model receives centered Cartesian positions;
the other receives the equivalent centered fractional positions plus the lattice.
Their losses should stay numerically close across training.
"""

import argparse
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from platonic_transformers.datasets.qm9 import QM9Dataset, collate_fn
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.lattice import cartesian_to_fractional
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.models.platoformer.utils import scatter_add


TARGETS = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atom",
    "U_atom",
    "H_atom",
    "G_atom",
    "A",
    "B",
    "C",
]


def hexagonal_lattice(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = 2.46
    c = 6.70
    return torch.tensor(
        [
            [a, 0.0, 0.0],
            [-0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=dtype,
        device=device,
    )


def center_by_graph(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1
    counts = scatter_add(torch.ones_like(pos[:, :1]), batch, dim_size=num_graphs).clamp_min(1.0)
    mean = scatter_add(pos, batch, dim_size=num_graphs) / counts
    return pos - mean[batch]


def target_from_batch(batch, target_idx: int) -> torch.Tensor:
    y = batch["y"] if isinstance(batch, dict) else batch.y
    if y.ndim == 2:
        return y[:, target_idx]
    return y.view(-1)


def dataset_stats(dataset: Subset, target_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ys = []
    num_nodes = []
    for data in dataset:
        y = data.y if hasattr(data, "y") else data["y"]
        ys.append(y.view(-1)[target_idx])
        num_nodes.append(data.num_nodes if hasattr(data, "num_nodes") else data["num_atoms"])
    y = torch.stack(ys).float()
    shift = y.mean()
    scale = y.std(unbiased=False).clamp_min(1e-12)
    avg_num_nodes = torch.tensor(float(sum(num_nodes)) / len(num_nodes), dtype=torch.float32)
    return shift, scale, avg_num_nodes


def build_model(args: argparse.Namespace, input_dim: int) -> PlatonicTransformer:
    solid_name = args.solid_name.lower()
    if solid_name not in PLATONIC_GROUPS:
        raise ValueError(f"Unknown solid_name {solid_name!r}; choices are {list(PLATONIC_GROUPS)}")
    group_size = PLATONIC_GROUPS[solid_name].G
    if args.hidden_dim % group_size != 0:
        raise ValueError(f"hidden_dim must be divisible by group size {group_size}.")
    if args.num_heads % group_size != 0:
        raise ValueError(f"num_heads must be divisible by group size {group_size}.")

    return PlatonicTransformer(
        input_dim=input_dim,
        input_dim_vec=0,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        output_dim_vec=0,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        solid_name=solid_name,
        spatial_dim=3,
        dense_mode=False,
        scalar_task_level="graph",
        vector_task_level="graph",
        ffn_readout=True,
        mean_aggregation=False,
        dropout=0.0,
        drop_path_rate=0.0,
        attention=True,
        ffn_dim_factor=2,
        rope_sigma=args.rope_sigma,
        ape_sigma=args.ape_sigma,
        learned_freqs=True,
        freq_init="spiral",
        use_key=False,
        rope_on_values=True,
        attention_backend="scatter",
    )


def load_qm9(args: argparse.Namespace) -> QM9Dataset:
    sdf_file = "gdb9.sdf"
    csv_file = "gdb9.sdf.csv"
    raw_sdf = os.path.join(args.data_dir, "raw", sdf_file)
    raw_csv = os.path.join(args.data_dir, "raw", csv_file)
    if os.path.exists(raw_sdf) and os.path.exists(raw_csv):
        sdf_file = os.path.join("raw", sdf_file)
        csv_file = os.path.join("raw", csv_file)
    return QM9Dataset(root=args.data_dir, sdf_file=sdf_file, csv_file=csv_file)


def train_pair(args: argparse.Namespace) -> tuple[float, float, float, float]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    full_dataset = load_qm9(args)
    target_idx = TARGETS.index(args.target)

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full_dataset), generator=generator)
    train_size = min(args.train_size, len(full_dataset))
    subset = Subset(full_dataset, perm[:train_size].tolist())
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    shift, scale, avg_num_nodes = dataset_stats(subset, target_idx)
    shift = shift.to(device)
    scale = scale.to(device)
    avg_num_nodes = avg_num_nodes.to(device)

    input_dim = full_dataset[0]["x"].shape[-1]
    cart_model = build_model(args, input_dim).to(device)
    frac_model = build_model(args, input_dim).to(device)
    frac_model.load_state_dict(cart_model.state_dict())

    cart_opt = torch.optim.AdamW(cart_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    frac_opt = torch.optim.AdamW(frac_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lattice_base = hexagonal_lattice(device, torch.float32)
    final_cart_loss = 0.0
    final_frac_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        cart_model.train()
        frac_model.train()
        cart_total = 0.0
        frac_total = 0.0
        graph_total = 0
        max_batch_gap = 0.0

        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            x = batch["x"].float()
            batch_idx = batch["batch"]
            num_graphs = int(batch_idx.max().item()) + 1
            cart_pos = center_by_graph(batch["pos"].float(), batch_idx)
            lattice = lattice_base.expand(num_graphs, -1, -1).contiguous()
            frac_pos = cartesian_to_fractional(cart_pos, lattice, batch=batch_idx)
            y = (target_from_batch(batch, target_idx).float() - shift) / scale

            cart_opt.zero_grad(set_to_none=True)
            frac_opt.zero_grad(set_to_none=True)

            cart_pred, _ = cart_model(x, cart_pos, batch_idx, avg_num_nodes=avg_num_nodes)
            frac_pred, _ = frac_model(
                x,
                frac_pos,
                batch_idx,
                lattice=lattice,
                fractional_pos=True,
                avg_num_nodes=avg_num_nodes,
            )
            cart_loss = F.l1_loss(cart_pred.squeeze(-1), y)
            frac_loss = F.l1_loss(frac_pred.squeeze(-1), y)

            cart_loss.backward()
            frac_loss.backward()
            cart_opt.step()
            frac_opt.step()

            batch_graphs = y.numel()
            cart_total += cart_loss.item() * batch_graphs
            frac_total += frac_loss.item() * batch_graphs
            graph_total += batch_graphs
            max_batch_gap = max(max_batch_gap, abs(cart_loss.item() - frac_loss.item()))

        final_cart_loss = cart_total / graph_total
        final_frac_loss = frac_total / graph_total
        epoch_gap = abs(final_cart_loss - final_frac_loss)
        print(
            f"epoch {epoch:02d}  "
            f"cart_loss={final_cart_loss:.8f}  "
            f"frac_loss={final_frac_loss:.8f}  "
            f"gap={epoch_gap:.3e}  "
            f"max_batch_gap={max_batch_gap:.3e}"
        )

    final_gap = abs(final_cart_loss - final_frac_loss)
    final_rel = final_gap / max(abs(final_cart_loss), abs(final_frac_loss), 1e-12)
    return final_cart_loss, final_frac_loss, final_gap, final_rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/qm9")
    parser.add_argument("--target", default="alpha", choices=TARGETS)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--solid-name", default="tetrahedron")
    parser.add_argument("--rope-sigma", type=float, default=1.5)
    parser.add_argument("--ape-sigma", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    cart_loss, frac_loss, gap, rel = train_pair(args)
    print(
        "final  "
        f"cart_loss={cart_loss:.8f}  "
        f"frac_loss={frac_loss:.8f}  "
        f"gap={gap:.3e}  "
        f"rel_gap={rel:.3e}"
    )
    if gap > args.atol and rel > args.rtol:
        raise SystemExit(
            f"fractional+lattice loss diverged from Cartesian loss: "
            f"gap={gap:.3e}, rel_gap={rel:.3e}"
        )
    print("QM9 lattice fractional/cartesian training check passed")


if __name__ == "__main__":
    main()
