import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from platonic_transformers.datasets.mp20 import (
    MP20CIFDataset,
    collate_crystal_batch,
    split_dataset,
)
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer


def default_data_dir() -> str:
    local_scratch = Path("/data/adeesh/crystal-llm-v2/cif_data")
    if local_scratch.exists():
        return str(local_scratch)
    return "data/mp20/cif_data"


def build_model(args: argparse.Namespace) -> PlatonicTransformer:
    return PlatonicTransformer(
        input_dim=args.max_atomic_number,
        input_dim_vec=0,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        output_dim_vec=0,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        solid_name=args.solid_name,
        scalar_task_level="graph",
        vector_task_level="graph",
        ffn_readout=True,
        mean_aggregation=True,
        attention=True,
        dropout=0.0,
        rope_sigma=args.rope_sigma,
        ape_sigma=None,
        learned_freqs=False,
        freq_init="spiral",
        rope_on_values=True,
        lattice_rope_mode="reciprocal",
    )


def normalize_targets(train_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    ys = []
    for batch in train_loader:
        ys.append(batch.y)
    y = torch.cat(ys)
    shift = y.mean()
    scale = y.std().clamp_min(1e-6)
    return shift, scale


def run_epoch(
    model: PlatonicTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    shift: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mae = 0.0
    total_graphs = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = batch.to(device)
        target = (batch.y - shift.to(device)) / scale.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        pred, _ = model(
            batch.x,
            batch.pos,
            batch=batch.batch,
            lattice=batch.cell,
            pbc=batch.pbc,
        )
        pred = pred.view(-1)
        loss = F.mse_loss(pred, target)
        if training:
            loss.backward()
            optimizer.step()

        graphs = batch.y.numel()
        total_graphs += graphs
        total_loss += loss.detach().item() * graphs
        total_mae += ((pred.detach() * scale.to(device) + shift.to(device)) - batch.y).abs().sum().item()

    return total_loss / total_graphs, total_mae / total_graphs


@torch.no_grad()
def periodic_shift_check(
    model: PlatonicTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    batch = next(iter(loader)).to(device)
    shifted = batch.pos.clone()
    for graph_idx in range(batch.cell.shape[0]):
        node_idx = torch.nonzero(batch.batch == graph_idx, as_tuple=False).flatten()
        if len(node_idx) > 0:
            shifted[node_idx[0]] += batch.cell[graph_idx, 0]

    pred, _ = model(batch.x, batch.pos, batch=batch.batch, lattice=batch.cell, pbc=batch.pbc)
    pred_shifted, _ = model(batch.x, shifted, batch=batch.batch, lattice=batch.cell, pbc=batch.pbc)
    return (pred - pred_shifted).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny MP20/CIF periodic smoke training run.")
    parser.add_argument("--data-dir", default=default_data_dir())
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-atoms", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--target", default="log_volume_per_atom")
    parser.add_argument("--parser", default="fast", choices=["fast", "pymatgen"])
    parser.add_argument("--max-atomic-number", type=int, default=118)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--solid-name", default="tetrahedron")
    parser.add_argument("--rope-sigma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = MP20CIFDataset(
        args.data_dir,
        limit=args.limit,
        max_atoms=args.max_atoms,
        target=args.target,
        max_atomic_number=args.max_atomic_number,
        parser=args.parser,
    )
    train_dataset, val_dataset = split_dataset(dataset, val_fraction=0.2, seed=args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_crystal_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystal_batch,
    )

    stats_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystal_batch,
    )
    shift, scale = normalize_targets(stats_loader)

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    print(
        f"MP20 smoke run: {len(train_dataset)} train / {len(val_dataset)} val, "
        f"target={args.target}, device={device}, shift={shift.item():.4f}, "
        f"scale={scale.item():.4f}, max_atoms={args.max_atoms}, parser={args.parser}, "
        "lattice_rope_mode=reciprocal",
        flush=True,
    )
    pre_shift_err = periodic_shift_check(model, val_loader, device)
    print(f"periodic shift max diff before training: {pre_shift_err:.3e}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(
            model,
            train_loader,
            optimizer,
            shift,
            scale,
            device,
            max_batches=args.max_batches,
        )
        with torch.no_grad():
            val_loss, val_mae = run_epoch(
                model,
                val_loader,
                None,
                shift,
                scale,
                device,
                max_batches=args.max_batches,
            )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"train_mae={train_mae:.4f} val_loss={val_loss:.4f} val_mae={val_mae:.4f}",
            flush=True,
        )

    post_shift_err = periodic_shift_check(model, val_loader, device)
    print(f"periodic shift max diff after training: {post_shift_err:.3e}", flush=True)


if __name__ == "__main__":
    main()
