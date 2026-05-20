"""Memmap-backed OMol25 datamodule — faster alternative to the LMDB path.

Reads per-field memmap arrays produced by scripts/preprocess_memmap.py.
Per-sample access is O(1) numpy slicing — no ASE deserialization, no
pickle, no LMDB get. Fits entirely in 737 GB node RAM (~150 GB total
for OMol25 train); OS page cache holds the whole thing after first
touch.

Swappable with OMol4mModule via yaml config (data=omol_full_memmap).
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from src.utils.file_utils import load_reference
from src.data.omol_4m_module import DynamicAtomBatchSamplerForAseDB


log = logging.getLogger(__name__)


class _MemmapArrays:
    """Lazy-loaded per-field memmaps for one split directory.

    We memmap in each worker process — the initial DataLoader worker
    spawn via fork should share underlying OS page cache so we don't
    pay RAM multiple times. r+ reopen is cheap.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.N = int(manifest["N"])
        self.total_atoms = int(manifest["total_atoms"])
        self.natoms_max = int(manifest["natoms_max"])
        self._open = False

    def _open_memmaps(self):
        if self._open:
            return
        p = self.path
        N = self.N
        T = self.total_atoms
        self.offsets = np.memmap(p / "offsets.bin", dtype="int64", mode="r", shape=(N + 1,))
        self.atomic_numbers = np.memmap(p / "atomic_numbers.bin", dtype="uint8", mode="r", shape=(T,))
        self.positions = np.memmap(p / "positions.bin", dtype="float32", mode="r", shape=(T, 3))
        self.forces = np.memmap(p / "forces.bin", dtype="float32", mode="r", shape=(T, 3))
        self.energy = np.memmap(p / "energy.bin", dtype="float32", mode="r", shape=(N,))
        self.charge = np.memmap(p / "charge.bin", dtype="int8", mode="r", shape=(N,))
        self.spin = np.memmap(p / "spin.bin", dtype="int8", mode="r", shape=(N,))
        self._open = True

    def __getstate__(self):
        # Don't pickle the memmap objects; they'll be reopened in the worker.
        return {"path": self.path, "N": self.N, "total_atoms": self.total_atoms, "natoms_max": self.natoms_max, "_open": False}

    def __setstate__(self, state):
        self.__dict__.update(state)


class MemmapAtomsDataset(Dataset):
    """Memmap-backed dataset returning fairchem AtomicData objects."""

    def __init__(self, memmap_dir, precompute_reference_energy=False, reference_path=None):
        self.arrays = _MemmapArrays(Path(memmap_dir))
        self.precompute_reference_energy = bool(precompute_reference_energy and reference_path)
        if self.precompute_reference_energy:
            refs = load_reference(reference_path).element_references.detach().to(dtype=torch.float32, device="cpu")
            self._element_references = refs
        else:
            self._element_references = None

    def __len__(self):
        return self.arrays.N

    def get_num_atoms(self, idx):
        self.arrays._open_memmaps()
        return int(self.arrays.offsets[idx + 1] - self.arrays.offsets[idx])

    def __getitem__(self, idx):
        self.arrays._open_memmaps()
        a = int(self.arrays.offsets[idx])
        b = int(self.arrays.offsets[idx + 1])
        n = b - a

        # Slice from memmap → np.ndarray. np.array(..., copy=True) both copies
        # out of the read-only mmap region and casts to the target dtype, so
        # torch.from_numpy receives a writable buffer.
        an = np.array(self.arrays.atomic_numbers[a:b], dtype=np.int64, copy=True)
        pos = np.array(self.arrays.positions[a:b], dtype=np.float32, copy=True)
        frc = np.array(self.arrays.forces[a:b], dtype=np.float32, copy=True)
        energy_scalar = float(self.arrays.energy[idx])
        charge_scalar = int(self.arrays.charge[idx])
        spin_scalar = int(self.arrays.spin[idx])

        # Build AtomicData. cell/pbc are irrelevant for non-periodic molecules;
        # edge_index/cell_offsets are empty because graph_scattered_attention
        # constructs edges on the GPU at each step. AtomicData.validate()
        # asserts pos.dtype == cell.dtype == cell_offsets.dtype → all float32.
        data = AtomicData(
            pos=torch.from_numpy(pos),
            atomic_numbers=torch.from_numpy(an),
            cell=torch.zeros(1, 3, 3, dtype=torch.float32),
            pbc=torch.zeros(1, 3, dtype=torch.bool),
            natoms=torch.tensor([n], dtype=torch.long),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            cell_offsets=torch.zeros(0, 3, dtype=torch.float32),
            nedges=torch.tensor([0], dtype=torch.long),
            charge=torch.tensor([charge_scalar], dtype=torch.long),
            spin=torch.tensor([spin_scalar], dtype=torch.long),
            fixed=torch.zeros(n, dtype=torch.long),
            tags=torch.zeros(n, dtype=torch.long),
            energy=torch.tensor([energy_scalar], dtype=torch.float32),
            forces=torch.from_numpy(frc),
        )

        if self.precompute_reference_energy and self._element_references is not None:
            refs = self._element_references
            valid = (data.atomic_numbers >= 0) & (data.atomic_numbers < refs.numel())
            ref_sum = refs[data.atomic_numbers[valid]].sum() if valid.any() else refs.new_tensor(0.0)
            corrected = data.energy.view(-1)[0] - ref_sum
            data.energy_ref_corrected = torch.full_like(data.energy, float(corrected.item()))

        return data


class OMolMemmapModule(L.LightningDataModule):
    """LightningDataModule backed by memmap arrays instead of LMDB."""

    def __init__(
        self,
        data,
        batch_size,
        dynamic_batching=True,
        max_atoms_per_batch=None,
        max_atoms_per_batch_val=None,
        max_edges_per_batch=None,
        max_edges_per_batch_val=None,
        prefetch_factor=2,
        debug_subset=None,
        validation_mode="heldout",
        train_size=0.9,
        precompute_reference_energy=True,
        reference_path=None,
    ):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.dynamic_batching = dynamic_batching
        self.max_atoms_per_batch = max_atoms_per_batch
        self.max_atoms_per_batch_val = max_atoms_per_batch_val or max_atoms_per_batch
        self.max_edges_per_batch = max_edges_per_batch
        self.max_edges_per_batch_val = max_edges_per_batch_val or max_edges_per_batch
        self.prefetch_factor = prefetch_factor
        self.debug_subset = debug_subset
        self.validation_mode = validation_mode
        self.train_size = train_size
        self.datasets = {}
        self.save_hyperparameters()

    def setup(self, stage=None):
        if self.datasets:
            return
        self.datasets["train"] = MemmapAtomsDataset(
            self.hparams.data.train_data_path,
            precompute_reference_energy=self.hparams.precompute_reference_energy,
            reference_path=getattr(self.hparams, "reference_path", None),
        )
        self.datasets["val"] = MemmapAtomsDataset(
            self.hparams.data.val_data_path,
            precompute_reference_energy=self.hparams.precompute_reference_energy,
            reference_path=getattr(self.hparams, "reference_path", None),
        )
        self.datasets["test"] = self.datasets["val"]
        log.info("# Training: %s  # Val: %s", len(self.datasets["train"]), len(self.datasets["val"]))

    def _create_dataloader(self, dataset, batch_size, shuffle):
        if self.dynamic_batching and self.max_atoms_per_batch is not None:
            max_atoms = self.max_atoms_per_batch if shuffle else self.max_atoms_per_batch_val
            max_edges = self.max_edges_per_batch if shuffle else self.max_edges_per_batch_val
            sampler = DynamicAtomBatchSamplerForAseDB(
                dataset,
                max_batch_size=999999,
                max_atoms=max_atoms,
                max_edges=max_edges,
                shuffle=shuffle,
                drop_last=False,
                seed=self.hparams.data.seed,
            )
            pf_kwargs = {}
            if self.hparams.data.num_workers > 0 and self.prefetch_factor:
                pf_kwargs["prefetch_factor"] = int(self.prefetch_factor)
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.hparams.data.num_workers,
                pin_memory=self.hparams.data.pin_memory,
                collate_fn=atomicdata_list_to_batch,
                persistent_workers=self.hparams.data.num_workers > 0,
                **pf_kwargs,
            )
            loader.set_epoch = sampler.set_epoch
            return loader
        pf_kwargs = {}
        if self.hparams.data.num_workers > 0 and self.prefetch_factor:
            pf_kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.data.num_workers,
            pin_memory=self.hparams.data.pin_memory,
            collate_fn=atomicdata_list_to_batch,
            persistent_workers=self.hparams.data.num_workers > 0,
            **pf_kwargs,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.datasets["train"], self.batch_size.train, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.datasets["val"], self.batch_size.val, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.datasets["test"], self.batch_size.val, shuffle=False)
