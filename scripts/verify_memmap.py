"""Verify memmap dataset by spot-checking samples against the LMDB source.

For N random sample indices, compare memmap output to fresh LMDB read.
Catches offset / striding / dtype bugs before we waste H100 time.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.data.omol_memmap_module import MemmapAtomsDataset


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: python verify_memmap.py <LMDB_SRC> <MEMMAP_DIR> [num_samples]")
        sys.exit(2)
    lmdb_src = Path(sys.argv[1])
    memmap_dir = Path(sys.argv[2])
    num_samples = int(sys.argv[3]) if len(sys.argv) >= 4 else 20

    from fairchem.core.datasets import AseDBDataset
    lmdb_ds = AseDBDataset({"src": str(lmdb_src), "a2g_args": dict(r_energy=True, r_forces=True)})
    memmap_ds = MemmapAtomsDataset(memmap_dir, precompute_reference_energy=False)

    print(f"lmdb len={len(lmdb_ds):,}  memmap len={len(memmap_ds):,}")
    assert len(lmdb_ds) == len(memmap_ds), "length mismatch"

    rng = np.random.default_rng(42)
    indices = rng.choice(len(memmap_ds), size=num_samples, replace=False)

    t_lmdb = t_memmap = 0.0
    for k, idx in enumerate(indices):
        idx = int(idx)
        t0 = time.time()
        lmdb_ad = lmdb_ds[idx]
        lmdb_atoms = lmdb_ds.get_atoms(idx)
        t_lmdb += time.time() - t0
        charge_ref = int(lmdb_atoms.info.get("charge", 0))
        spin_ref = int(lmdb_atoms.info.get("spin", 0))

        t0 = time.time()
        mm_ad = memmap_ds[idx]
        t_memmap += time.time() - t0

        n_ref = lmdb_ad.atomic_numbers.shape[0]
        n_mm = mm_ad.atomic_numbers.shape[0]
        assert n_ref == n_mm, f"idx {idx}: natoms {n_ref} vs {n_mm}"

        an_ok = torch.equal(
            lmdb_ad.atomic_numbers.long().view(-1),
            mm_ad.atomic_numbers.long().view(-1),
        )
        pos_close = torch.allclose(
            lmdb_ad.pos.float(), mm_ad.pos.float(), atol=1e-5
        )
        frc_close = torch.allclose(
            lmdb_ad.forces.float(), mm_ad.forces.float(), atol=1e-5
        )
        e_ref = float(lmdb_ad.energy.view(-1)[0].item())
        e_mm = float(mm_ad.energy.view(-1)[0].item())
        e_close = abs(e_ref - e_mm) < max(1e-4 * abs(e_ref), 1e-3)
        c_ok = int(mm_ad.charge.view(-1)[0].item()) == charge_ref
        s_ok = int(mm_ad.spin.view(-1)[0].item()) == spin_ref

        ok = an_ok and pos_close and frc_close and e_close and c_ok and s_ok
        mark = "OK" if ok else "FAIL"
        print(f"[{k:>3}] idx={idx:>10}  n={n_ref:>3}  atomic={an_ok}  pos={pos_close}  forces={frc_close}  energy={e_close}  charge={c_ok}  spin={s_ok}  → {mark}")
        if not ok:
            print(f"      e_ref={e_ref}  e_mm={e_mm}")

    print(f"\nLMDB total: {t_lmdb:.3f}s  ({t_lmdb/num_samples*1000:.2f} ms/sample)")
    print(f"Memmap total: {t_memmap:.3f}s  ({t_memmap/num_samples*1000:.2f} ms/sample)")
    print(f"Speedup: {t_lmdb/max(t_memmap,1e-9):.1f}x")


if __name__ == "__main__":
    main()
