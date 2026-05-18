"""Measure raw dataloader throughput without any model forward/backward.

Usage: python bench_dataloader.py <dynamic|fixed> [max_atoms] [num_workers] [num_batches]

Prints: total atoms/s, total samples/s, sec/batch, atoms/batch for warmup +
two equal measurement windows (early vs late) to detect cache warm-up.
"""
from __future__ import annotations

import os
import sys
import time

from omegaconf import OmegaConf

# PYTHONPATH must include the training/ directory (set by run_bench_dataloader.sh).


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dynamic"
    max_atoms = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    num_batches = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    warmup = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    backend = sys.argv[6] if len(sys.argv) > 6 else "lmdb"  # lmdb | memmap
    split = sys.argv[7] if len(sys.argv) > 7 else "train"  # train | val

    if backend == "memmap":
        from src.data.omol_memmap_module import OMolMemmapModule as ModuleCls
        base = f"{os.environ['DATA_PATH']}/open_mol_memmap"
    else:
        from src.data.omol_4m_module import OMol4mModule as ModuleCls
        base = f"{os.environ['DATA_PATH']}/open_mol"

    # If split=val we point both train/val paths at val for this bench.
    split_dir_train = f"{base}/val" if split == "val" else f"{base}/train"
    split_dir_val = f"{base}/val"

    cfg = OmegaConf.create({
        "data": {
            "train_data_path": split_dir_train,
            "val_data_path": split_dir_val,
            "pin_memory": True,
            "num_workers": num_workers,
            "seed": 42,
            "train_size": 0.9,
            "shuffle": True,
        },
        "batch_size": {"train": 64, "val": 64},
    })

    dm = ModuleCls(
        data=cfg.data,
        batch_size=cfg.batch_size,
        dynamic_batching=(mode == "dynamic"),
        max_atoms_per_batch=max_atoms if mode == "dynamic" else None,
        max_atoms_per_batch_val=max_atoms if mode == "dynamic" else None,
        validation_mode="heldout",
        precompute_reference_energy=True,
        reference_path="configs/constants/element_refs.yaml",
    )

    t0 = time.time()
    dm.setup("fit")
    loader = dm.train_dataloader()
    print(f"[setup] {time.time()-t0:.1f}s  mode={mode}  max_atoms={max_atoms}  workers={num_workers}  num_batches={num_batches}  warmup={warmup}")

    it = iter(loader)

    # Warmup
    t0 = time.time()
    atoms_w = 0
    for i in range(warmup):
        batch = next(it)
        atoms_w += batch.atomic_numbers.shape[0]
    print(f"[warmup] {warmup} batches in {time.time()-t0:.2f}s  atoms={atoms_w}  tok/s={atoms_w/(time.time()-t0):.0f}")

    # Measurement — split into two halves to detect warmup/cache drift
    half = num_batches // 2
    windows = [("early", half), ("late", num_batches - half)]
    atoms_total = 0
    samples_total = 0
    t_start = time.time()
    for label, n in windows:
        t0 = time.time()
        atoms = 0
        samples = 0
        for i in range(n):
            batch = next(it)
            atoms += batch.atomic_numbers.shape[0]
            samples += len(batch.batch.unique()) if hasattr(batch, "batch") else 1
        dt = time.time() - t0
        atoms_total += atoms
        samples_total += samples
        print(f"[{label}] n={n} dt={dt:.2f}s tok/s={atoms/dt:.0f} samples/s={samples/dt:.1f} atoms/batch={atoms/n:.0f} sec/batch={dt/n:.3f}")

    total_dt = time.time() - t_start
    print(f"[total] n={num_batches} dt={total_dt:.2f}s tok/s={atoms_total/total_dt:.0f}")


if __name__ == "__main__":
    main()
