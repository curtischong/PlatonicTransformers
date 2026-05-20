"""Resume preprocess_memmap for a specific list of (lo, hi) chunk ranges.

The original run (Slurm 22055213) hung: 9 of 16 train workers never printed
a single line — diagnosis points to fork-inherited state making fairchem /
LMDB open silently deadlock. This script reprocesses a user-supplied list
of chunk ranges using the `spawn` start method (clean Python per worker).

The destination directory must already have manifest.json + pre-allocated
files (from a prior preprocess_memmap run). We open the memmaps in r+ and
overwrite the requested ranges.

Usage:
    python preprocess_memmap_resume.py <SRC_SPLIT_DIR> <DEST_DIR> \
        <RANGES>  [num_workers]

RANGES: comma-separated "lo-hi" pairs, e.g.
    6354142-12708284,12708284-19062427,...
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np


def _worker(args):
    lo, hi, src_str, dest_str = args
    pid = os.getpid()
    # Print immediately on entry so we see if later steps hang.
    print(f"[pid={pid}] start {lo}..{hi}", flush=True)

    src = Path(src_str)
    dest = Path(dest_str)

    from fairchem.core.datasets import AseDBDataset
    print(f"[pid={pid}] fairchem imported", flush=True)
    ds = AseDBDataset({"src": str(src), "a2g_args": dict(r_energy=True, r_forces=True)})
    print(f"[pid={pid}] AseDB opened ({len(ds)} samples)", flush=True)

    manifest = json.loads((dest / "manifest.json").read_text())
    total_atoms = manifest["total_atoms"]
    n = manifest["N"]
    offsets = np.memmap(dest / "offsets.bin", dtype="int64", mode="r", shape=(n + 1,))
    atomic_numbers = np.memmap(dest / "atomic_numbers.bin", dtype="uint8", mode="r+", shape=(total_atoms,))
    positions = np.memmap(dest / "positions.bin", dtype="float32", mode="r+", shape=(total_atoms, 3))
    forces = np.memmap(dest / "forces.bin", dtype="float32", mode="r+", shape=(total_atoms, 3))
    energy = np.memmap(dest / "energy.bin", dtype="float32", mode="r+", shape=(n,))
    charge = np.memmap(dest / "charge.bin", dtype="int8", mode="r+", shape=(n,))
    spin = np.memmap(dest / "spin.bin", dtype="int8", mode="r+", shape=(n,))
    print(f"[pid={pid}] memmaps opened r+", flush=True)

    t0 = time.time()
    report_every = max(1, (hi - lo) // 20)

    for i in range(lo, hi):
        a = offsets[i]
        b = offsets[i + 1]
        atomic_data = ds[i]
        atoms = ds.get_atoms(i)
        atomic_numbers[a:b] = np.asarray(atomic_data.atomic_numbers, dtype=np.uint8).reshape(-1)
        positions[a:b] = np.asarray(atomic_data.pos, dtype=np.float32).reshape(-1, 3)
        forces[a:b] = np.asarray(atomic_data.forces, dtype=np.float32).reshape(-1, 3)
        energy[i] = float(atomic_data.energy.view(-1)[0].item())
        charge[i] = int(atoms.info.get("charge", 0))
        spin[i] = int(atoms.info.get("spin", 0))

        if (i - lo) % report_every == 0:
            elapsed = time.time() - t0
            rate = (i - lo + 1) / max(elapsed, 1e-6)
            eta = (hi - i - 1) / max(rate, 1e-6)
            print(f"[pid={pid}] {i}/{hi}  rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

    for m in (atomic_numbers, positions, forces, energy, charge, spin):
        m.flush()
    print(f"[pid={pid}] done {lo}..{hi} in {time.time()-t0:.1f}s", flush=True)
    return lo, hi


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    ranges_str = sys.argv[3]
    num_workers = int(sys.argv[4]) if len(sys.argv) >= 5 else 9

    ranges = []
    for token in ranges_str.split(","):
        token = token.strip()
        if not token:
            continue
        lo_s, hi_s = token.split("-")
        ranges.append((int(lo_s), int(hi_s)))

    print(f"[src]  {src}")
    print(f"[dest] {dest}")
    print(f"[workers] {num_workers}")
    print(f"[ranges] {len(ranges)} chunks:")
    for lo, hi in ranges:
        print(f"         {lo:>10} .. {hi:>10}  ({hi-lo:,} samples)")

    tasks = [(lo, hi, str(src), str(dest)) for lo, hi in ranges]

    # `spawn` avoids inheriting any fork state from main — the original hang
    # was almost certainly a fork+fairchem/LMDB interaction.
    ctx = mp.get_context("spawn")
    print(f"[start_method] spawn", flush=True)

    t0 = time.time()
    with ctx.Pool(num_workers) as pool:
        for lo, hi in pool.imap_unordered(_worker, tasks):
            print(f"[main] chunk {lo}..{hi} returned", flush=True)
    print(f"[parallel] all workers done in {time.time()-t0:.1f}s")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
