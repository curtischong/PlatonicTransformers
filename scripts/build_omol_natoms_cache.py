#!/usr/bin/env python3
"""Build the `metadata.npz` atom-count cache next to an aselmdb shard directory.

The OMol25 release from fairchem ships `metadata.npz` next to each split's
`.aselmdb` shards, with a `natoms` (int64) array indexed the same as the
dataset. Our `DynamicAtomBatchSampler` uses this cache to look up atom counts
in O(1) when packing batches by atom budget. Without it, each lookup is a full
LMDB read in the main process — drastically slow on large datasets over shared
storage, to the point of making `dynamic_batching=true` unusable.

Use this script when you have a custom aselmdb shard directory (e.g. a
symlinked subset, or shards you generated yourself) and need to materialize
the cache. One-time cost: ~minutes for a 4M-sample dataset.

Usage:
    python scripts/build_omol_natoms_cache.py <aselmdb_dir>

After it finishes, `<aselmdb_dir>/metadata.npz` will contain the natoms array
keyed by index, and subsequent training runs with `dynamic_batching=true`
will pick it up automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("aselmdb_dir", type=str,
                        help="Directory containing .aselmdb shard files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing metadata.npz")
    args = parser.parse_args()

    src = os.path.abspath(args.aselmdb_dir)
    if not os.path.isdir(src):
        print(f"ERROR: {src} is not a directory", file=sys.stderr)
        return 1

    out_path = os.path.join(src, "metadata.npz")
    if os.path.exists(out_path) and not args.force:
        print(f"{out_path} already exists. Use --force to overwrite.")
        return 0

    # Lazy import so the help text works without the dep.
    from fairchem.core.datasets import AseDBDataset

    print(f"Opening AseDBDataset at {src} ...")
    dataset = AseDBDataset({"src": src})
    n = len(dataset)
    print(f"  {n} samples; reading atom counts ...")

    natoms = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i in range(n):
        natoms[i] = len(dataset.get_atoms(i))
        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / max(rate, 1.0)
            print(f"  {i + 1:>10,}/{n:,}  ({rate:6.0f} samples/s, ETA {eta / 60:5.1f} min)")

    np.savez(out_path, natoms=natoms)
    elapsed = time.time() - t0
    print(f"Wrote {out_path} ({n} entries, {elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
