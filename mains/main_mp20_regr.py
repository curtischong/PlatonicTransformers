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

from platonic_transformers.datasets.mp20 import MP20CSVRegressionDataset, collate_crystal_batch
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer


def default_csv_dir() -> str:
    local = Path("/data/adeesh/crystal-llm-v2/resources/benchmarks/mp_20")
    if local.exists():
        return str(local)
    return "data/mp20"


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
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        rope_sigma=args.rope_sigma,
        ape_sigma=None,
        learned_freqs=False,
        freq_init="spiral",
        rope_on_values=True,
        lattice_rope_mode=args.lattice_rope_mode,
    )


def make_loader(dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_crystal_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def dataset_stats(dataset: MP20CSVRegressionDataset) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.tensor(dataset.targets, dtype=torch.float32)
    return y.mean(), y.std(unbiased=False).clamp_min(1e-6)


def run_epoch(
    model: PlatonicTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    shift: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype | None,
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

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=(amp_dtype is not None and device.type == "cuda"),
        ):
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
            if max_batches is None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        graphs = batch.y.numel()
        total_graphs += graphs
        total_loss += loss.detach().float().item() * graphs
        pred_real = pred.detach().float() * scale.to(device) + shift.to(device)
        total_mae += (pred_real - batch.y).abs().sum().item()

    return total_loss / total_graphs, total_mae / total_graphs


@torch.no_grad()
def periodic_shift_check(
    model: PlatonicTransformer,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    model.eval()
    batch = next(iter(loader)).to(device)
    shifted = batch.pos.clone()
    for graph_idx in range(batch.cell.shape[0]):
        node_idx = torch.nonzero(batch.batch == graph_idx, as_tuple=False).flatten()
        if len(node_idx) > 0:
            shifted[node_idx[0]] += batch.cell[graph_idx, 0]

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=(amp_dtype is not None and device.type == "cuda"),
    ):
        pred, _ = model(batch.x, batch.pos, batch=batch.batch, lattice=batch.cell, pbc=batch.pbc)
        pred_shifted, _ = model(batch.x, shifted, batch=batch.batch, lattice=batch.cell, pbc=batch.pbc)
    return (pred.float() - pred_shifted.float()).abs().max().item()


def save_checkpoint(
    path: Path,
    model: PlatonicTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_mae: float,
    shift: torch.Tensor,
    scale: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_mae": best_val_mae,
            "shift": shift,
            "scale": scale,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PlatonicTransformer on MP20 CSV splits.")
    parser.add_argument("--csv-dir", default=default_csv_dir())
    parser.add_argument("--target", default="formation_energy_per_atom")
    parser.add_argument("--max-atomic-number", type=int, default=118)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=1152)
    parser.add_argument("--num-heads", type=int, default=72)
    parser.add_argument("--num-layers", type=int, default=14)
    parser.add_argument("--solid-name", default="tetrahedron")
    parser.add_argument("--rope-sigma", type=float, default=1.0)
    parser.add_argument("--lattice-rope-mode", default="reciprocal", choices=["reciprocal", "minimum_image"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--amp", default="bf16", choices=["none", "bf16", "fp16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/mp20_regr")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_dtype = None
    if device.type == "cuda" and args.amp == "bf16":
        amp_dtype = torch.bfloat16
    elif device.type == "cuda" and args.amp == "fp16":
        amp_dtype = torch.float16

    csv_dir = Path(args.csv_dir)
    train_dataset = MP20CSVRegressionDataset(
        csv_dir / "train.csv",
        target=args.target,
        max_atomic_number=args.max_atomic_number,
        limit=args.limit_train,
    )
    val_dataset = MP20CSVRegressionDataset(
        csv_dir / "val.csv",
        target=args.target,
        max_atomic_number=args.max_atomic_number,
        limit=args.limit_val,
    )
    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)

    shift, scale = dataset_stats(train_dataset)
    model = build_model(args).to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_mae = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        shift = ckpt["shift"]
        scale = ckpt["scale"]
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))

    print(
        f"MP20 full-data run: {len(train_dataset)} train / {len(val_dataset)} val, "
        f"target={args.target}, device={device}, amp={args.amp}, "
        f"hidden_dim={args.hidden_dim}, layers={args.num_layers}, heads={args.num_heads}, "
        f"shift={shift.item():.4f}, scale={scale.item():.4f}",
        flush=True,
    )
    shift_err = periodic_shift_check(model, val_loader, device, None)
    print(f"periodic shift max diff before training (fp32): {shift_err:.3e}", flush=True)
    if amp_dtype is not None:
        shift_err_amp = periodic_shift_check(model, val_loader, device, amp_dtype)
        print(
            f"periodic shift max diff before training ({args.amp}): {shift_err_amp:.3e}",
            flush=True,
        )

    ckpt_dir = Path(args.checkpoint_dir)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_mae = run_epoch(
            model, train_loader, optimizer, shift, scale, device, amp_dtype, args.max_batches
        )
        with torch.no_grad():
            val_loss, val_mae = run_epoch(
                model, val_loader, None, shift, scale, device, amp_dtype, args.max_batches
            )

        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.5f} train_mae={train_mae:.5f} "
            f"val_loss={val_loss:.5f} val_mae={val_mae:.5f}",
            flush=True,
        )

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_val_mae,
            shift,
            scale,
            args,
        )
        if improved:
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_mae,
                shift,
                scale,
                args,
            )

    shift_err = periodic_shift_check(model, val_loader, device, None)
    print(f"periodic shift max diff after training (fp32): {shift_err:.3e}", flush=True)
    if amp_dtype is not None:
        shift_err_amp = periodic_shift_check(model, val_loader, device, amp_dtype)
        print(
            f"periodic shift max diff after training ({args.amp}): {shift_err_amp:.3e}",
            flush=True,
        )
    print(f"best val MAE: {best_val_mae:.5f}", flush=True)


if __name__ == "__main__":
    main()
