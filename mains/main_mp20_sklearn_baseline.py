import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mains.main_mp20_regr import default_csv_dir
from platonic_transformers.datasets.mp20 import MP20CSVRegressionDataset


def item_to_features(item: dict[str, object], max_atomic_number: int = 118) -> np.ndarray:
    atomic_numbers = item["atomic_numbers"]
    cell = item["cell"]
    assert isinstance(atomic_numbers, torch.Tensor)
    assert isinstance(cell, torch.Tensor)

    valid = (atomic_numbers >= 1) & (atomic_numbers <= max_atomic_number)
    counts = torch.bincount(
        atomic_numbers[valid] - 1,
        minlength=max_atomic_number,
    ).to(torch.float32)
    fractions = counts / counts.sum().clamp_min(1.0)
    present = (counts > 0).to(torch.float32)

    lengths = torch.linalg.norm(cell.float(), dim=1).clamp_min(1e-8)
    cos_alpha = torch.dot(cell[1].float(), cell[2].float()) / (lengths[1] * lengths[2])
    cos_beta = torch.dot(cell[0].float(), cell[2].float()) / (lengths[0] * lengths[2])
    cos_gamma = torch.dot(cell[0].float(), cell[1].float()) / (lengths[0] * lengths[1])

    num_atoms = float(item["num_atoms"])
    volume = float(item["volume"])
    volume_per_atom = volume / max(num_atoms, 1.0)
    spacegroup = float(item.get("spacegroup_number", 0.0))
    geometry = torch.tensor(
        [
            num_atoms,
            math.log1p(num_atoms),
            volume,
            math.log(max(volume, 1e-8)),
            volume_per_atom,
            math.log(max(volume_per_atom, 1e-8)),
            lengths[0].item(),
            lengths[1].item(),
            lengths[2].item(),
            cos_alpha.item(),
            cos_beta.item(),
            cos_gamma.item(),
            spacegroup / 230.0,
        ],
        dtype=torch.float32,
    )

    return torch.cat([fractions, present, geometry]).numpy()


def load_split(
    csv_path: Path,
    target: str,
    max_atomic_number: int,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = MP20CSVRegressionDataset(
        csv_path,
        target=target,
        max_atomic_number=max_atomic_number,
        limit=limit,
        preload=False,
    )
    features = np.stack([item_to_features(item, max_atomic_number) for item in dataset])
    targets = np.asarray(dataset.targets, dtype=np.float32)
    return features, targets


def evaluate(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = model.predict(x)
    mae = mean_absolute_error(y, pred)
    mse = float(np.mean((pred - y) ** 2))
    return {
        "mae": float(mae),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Composition/lattice sklearn baseline for MP20 formation-energy regression."
    )
    parser.add_argument("--csv-dir", default=default_csv_dir())
    parser.add_argument("--target", default="formation_energy_per_atom")
    parser.add_argument("--max-atomic-number", type=int, default=118)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--model", default="extra_trees", choices=["extra_trees", "hist_gradient_boosting"])
    parser.add_argument("--n-estimators", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--loss", default="squared_error", choices=["squared_error", "absolute_error"])
    parser.add_argument("--checkpoint-dir", default="checkpoints/mp20_sklearn_baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    start_time = time.time()
    csv_dir = Path(args.csv_dir)
    print("loading MP20 feature matrices", flush=True)
    x_train, y_train = load_split(
        csv_dir / "train.csv",
        args.target,
        args.max_atomic_number,
        args.limit_train,
    )
    x_val, y_val = load_split(
        csv_dir / "val.csv",
        args.target,
        args.max_atomic_number,
        args.limit_val,
    )
    x_test, y_test = load_split(
        csv_dir / "test.csv",
        args.target,
        args.max_atomic_number,
        args.limit_test,
    )

    if args.model == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            loss=args.loss,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2,
            random_state=args.seed,
            early_stopping=False,
        )
    print(
        f"training sklearn baseline: {len(y_train)} train / {len(y_val)} val / {len(y_test)} test, "
        f"features={x_train.shape[1]}",
        flush=True,
    )
    fit_start_time = time.time()
    model.fit(x_train, y_train)
    fit_seconds = time.time() - fit_start_time

    metrics = {
        "args": vars(args),
        "dataset": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
            "target": args.target,
            "features": int(x_train.shape[1]),
        },
        "train": evaluate(model, x_train, y_train),
        "val": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
        "runtime": {
            "fit_seconds": fit_seconds,
            "total_seconds": time.time() - start_time,
        },
    }

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, checkpoint_dir / "model.joblib")
    (checkpoint_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"val_mae={metrics['val']['mae']:.5f} test_mae={metrics['test']['mae']:.5f} "
        f"fit_time={fit_seconds / 60.0:.1f} min total_time={metrics['runtime']['total_seconds'] / 60.0:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
