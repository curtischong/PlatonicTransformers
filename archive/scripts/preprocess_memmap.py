"""One-time preprocessing: OMol25 LMDB → flat memmap arrays.

Converts the ASE LMDB dataset into a per-field flat-memmap layout that
can be memmapped at training time with ~zero deserialization cost. The
resulting files fit entirely in RAM (~150 GB vs 527 GB LMDB), so every
sample read is a direct slice of a memmap.

Layout (written under DEST/train/ and DEST/val/):
  offsets.bin       int64   [N+1]            cumulative atom offsets
  atomic_numbers.bin uint8   [total_atoms]
  positions.bin     float32 [total_atoms, 3]
  forces.bin        float32 [total_atoms, 3]
  energy.bin        float32 [N]
  charge.bin        int8    [N]
  spin.bin          int8    [N]
  manifest.json                                {N, total_atoms, dtype, fields}

Invoke with:
    python preprocess_memmap.py <SRC_SPLIT_DIR> <DEST_DIR> [num_workers]

  SRC_SPLIT_DIR  e.g. /scratch-shared/ebekkers/omol25/open_mol/train
  DEST_DIR       e.g. /scratch-shared/ebekkers/omol25/open_mol_memmap/train
  num_workers    default 16

Parallelism: workers take disjoint [lo, hi) index ranges. Since each
sample's target offset is known up-front from natoms, workers write to
non-overlapping memmap regions without locks.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np


def _load_natoms(src: Path) -> np.ndarray:
    meta = np.load(src / "metadata.npz", allow_pickle=False)
    natoms = np.asarray(meta["natoms"], dtype=np.int64)
    return natoms


def _preallocate(dest: Path, natoms: np.ndarray) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    n = int(natoms.shape[0])
    total_atoms = int(natoms.sum())

    # Offsets[i] = start index in flat atom arrays for sample i. Offsets[N] = total_atoms.
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(natoms, out=offsets[1:])
    offsets.tofile(dest / "offsets.bin")

    # Pre-allocate flat arrays (sparse — filled in parallel workers).
    def alloc(name: str, shape, dtype):
        path = dest / f"{name}.bin"
        # Write a single zero byte at the last position to create a sparse file
        # of the right size; workers fill it in via memmap 'r+'.
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        with open(path, "wb") as f:
            f.seek(size - 1)
            f.write(b"\0")
        return path, shape, dtype

    info = {
        "N": n,
        "total_atoms": total_atoms,
        "natoms_min": int(natoms.min()),
        "natoms_max": int(natoms.max()),
        "natoms_mean": float(natoms.mean()),
        "fields": {},
    }
    info["fields"]["offsets"] = {"shape": [n + 1], "dtype": "int64"}
    for name, shape, dtype in [
        ("atomic_numbers", (total_atoms,), "uint8"),
        ("positions", (total_atoms, 3), "float32"),
        ("forces", (total_atoms, 3), "float32"),
        ("energy", (n,), "float32"),
        ("charge", (n,), "int8"),
        ("spin", (n,), "int8"),
    ]:
        alloc(name, shape, dtype)
        info["fields"][name] = {"shape": list(shape), "dtype": dtype}

    (dest / "manifest.json").write_text(json.dumps(info, indent=2))
    return info


def _worker(args):
    lo, hi, src_str, dest_str, offsets_lo = args
    src = Path(src_str)
    dest = Path(dest_str)

    # Local imports inside worker (fairchem imports are heavy).
    from fairchem.core.datasets import AseDBDataset
    ds = AseDBDataset({"src": str(src), "a2g_args": dict(r_energy=True, r_forces=True)})

    # Open per-field memmaps in r+ mode for writing.
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

    t0 = time.time()
    report_every = max(1, (hi - lo) // 20)

    for i in range(lo, hi):
        a = offsets[i]
        b = offsets[i + 1]
        # atomic_data has tensors for atomic_numbers, pos, forces, energy
        atomic_data = ds[i]
        # ase Atoms object holds charge/spin in .info
        atoms = ds.get_atoms(i)

        # Cast and write
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
            print(f"[pid={os.getpid()}] {i}/{hi}  rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)

    # Flush memmaps on worker exit.
    for m in (atomic_numbers, positions, forces, energy, charge, spin):
        m.flush()
    print(f"[pid={os.getpid()}] done {lo}..{hi} in {time.time()-t0:.1f}s", flush=True)
    return lo, hi


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    num_workers = int(sys.argv[3]) if len(sys.argv) >= 4 else 16

    print(f"[src] {src}")
    print(f"[dest] {dest}")
    print(f"[workers] {num_workers}")

    t0 = time.time()
    natoms = _load_natoms(src)
    print(f"[load natoms] {natoms.shape[0]:,} samples, total {natoms.sum():,} atoms in {time.time()-t0:.1f}s")

    t0 = time.time()
    info = _preallocate(dest, natoms)
    print(f"[preallocate] offsets + 6 flat arrays in {time.time()-t0:.1f}s")

    n = info["N"]
    # Split indices into num_workers chunks.
    edges = np.linspace(0, n, num_workers + 1, dtype=np.int64)
    tasks = []
    offsets_arr = np.memmap(dest / "offsets.bin", dtype="int64", mode="r", shape=(n + 1,))
    for k in range(num_workers):
        lo, hi = int(edges[k]), int(edges[k + 1])
        tasks.append((lo, hi, str(src), str(dest), int(offsets_arr[lo])))

    print(f"[parallel] launching {num_workers} workers, {n:,} samples total")
    t0 = time.time()
    with mp.Pool(num_workers) as pool:
        for lo, hi in pool.imap_unordered(_worker, tasks):
            print(f"[main] chunk {lo}..{hi} returned", flush=True)
    print(f"[parallel] all workers done in {time.time()-t0:.1f}s")

    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
