import logging
import math
import os

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from fairchem.core.datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from src.utils.file_utils import load_reference


log = logging.getLogger(__name__)


class AseDBDatasetWithChargeSpin(Dataset):
    """FairChem AseDBDataset wrapper that injects graph-level charge and spin."""

    def __init__(self, config, precompute_reference_energy=False, reference_path=None):
        self.config = dict(config)
        self.base_dataset = AseDBDataset(self.config)
        self.precompute_reference_energy = bool(precompute_reference_energy and reference_path)
        self.reference_path = reference_path
        self._energy_ref_cache = None

        if self.precompute_reference_energy:
            refs = load_reference(reference_path).element_references.detach().to(dtype=torch.float32, device="cpu")
            self._element_references = refs
        else:
            self._element_references = None

        # Load pre-computed per-sample atom counts from metadata.npz if present.
        # OMol25 ships a `metadata.npz` next to the aselmdb shards with arrays
        # `natoms` and `data_ids` indexed the same as the dataset. Using this
        # cache turns `get_num_atoms(idx)` into an O(1) numpy lookup — crucial
        # for the DynamicAtomBatchSampler which otherwise does one full LMDB
        # get per sample just to peek the atom count (drastically slow on a
        # cold 527 GB dataset over shared GPFS).
        self._natoms_cache = None
        try:
            src = self.config.get("src")
            if isinstance(src, (list, tuple)):
                # Multiple sources not currently supported by this cache path.
                # AseDBDataset itself supports a list of paths, but metadata.npz
                # would then need to be concatenated in the same order.
                src_path = None
            else:
                src_path = src
            if src_path:
                metadata_path = os.path.join(src_path, "metadata.npz")
                if os.path.exists(metadata_path):
                    with np.load(metadata_path, allow_pickle=False) as m:
                        if "natoms" in m.files:
                            natoms = np.asarray(m["natoms"], dtype=np.int64)
                            if len(natoms) == len(self.base_dataset):
                                self._natoms_cache = natoms
                                log.info(
                                    "natoms cache loaded from %s (%d samples)",
                                    metadata_path, len(natoms),
                                )
                            else:
                                log.warning(
                                    "metadata.npz natoms length %d != dataset len %d at %s; "
                                    "ignoring cache.", len(natoms), len(self.base_dataset), src_path,
                                )
        except Exception as exc:  # pragma: no cover — defensive I/O
            log.warning("Failed to load natoms from metadata.npz: %s", exc)
            self._natoms_cache = None

    def _get_base_dataset(self):
        if self.base_dataset is None:
            self.base_dataset = AseDBDataset(self.config)
        return self.base_dataset

    def __getstate__(self):
        state = self.__dict__.copy()
        # AseDBDataset keeps unpicklable DB handles; reopen lazily in child process.
        state["base_dataset"] = None
        # Keep cache process-local to avoid large pickles in dataloader workers.
        state["_energy_ref_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self._get_base_dataset())

    def _ensure_energy_ref_cache(self):
        if not self.precompute_reference_energy:
            return None
        if self._energy_ref_cache is None:
            self._energy_ref_cache = torch.full((len(self._get_base_dataset()),), torch.nan, dtype=torch.float32)
        return self._energy_ref_cache

    def _compute_corrected_energy_value(self, atomic_numbers, energy):
        refs = self._element_references
        if refs is None:
            return float(energy.view(-1)[0].item())

        atomic_numbers = atomic_numbers.view(-1).to(dtype=torch.long, device=refs.device)
        valid = (atomic_numbers >= 0) & (atomic_numbers < refs.numel())
        if valid.any():
            ref_sum = refs[atomic_numbers[valid]].sum()
        else:
            ref_sum = refs.new_tensor(0.0)

        corrected = energy.view(-1)[0].to(dtype=refs.dtype, device=refs.device) - ref_sum
        return float(corrected.item())

    def _get_or_compute_corrected_energy_value(self, idx, atomic_numbers, energy):
        cache = self._ensure_energy_ref_cache()
        if cache is not None:
            cached = cache[idx]
            if not torch.isnan(cached):
                return float(cached.item())

        corrected_value = self._compute_corrected_energy_value(atomic_numbers, energy)

        if cache is not None:
            cache[idx] = corrected_value

        return corrected_value

    def __getitem__(self, idx):
        base_dataset = self._get_base_dataset()
        atomic_data = base_dataset[idx]
        atoms = base_dataset.get_atoms(idx)

        charge = int(atoms.info.get("charge", 0))
        spin = int(atoms.info.get("spin", 0))

        atomic_data.charge = torch.tensor([charge], dtype=torch.long)
        atomic_data.spin = torch.tensor([spin], dtype=torch.long)

        if self.precompute_reference_energy and getattr(atomic_data, "energy", None) is not None:
            corrected_value = self._get_or_compute_corrected_energy_value(
                idx=idx,
                atomic_numbers=atomic_data.atomic_numbers,
                energy=atomic_data.energy,
            )
            atomic_data.energy_ref_corrected = torch.full_like(atomic_data.energy, corrected_value)

        return atomic_data

    def get_num_atoms(self, idx):
        # O(1) lookup via the pre-loaded natoms array when metadata.npz is
        # available. Falls back to the slow LMDB path only when the cache
        # isn't there (e.g. synthetic datasets or non-OMol25 sources).
        if self._natoms_cache is not None:
            return int(self._natoms_cache[int(idx)])
        return len(self._get_base_dataset().get_atoms(idx))


class DynamicAtomBatchSamplerForAseDB(Sampler):
    """Batch sampler that caps graph count, total atoms, and (optionally) total
    within-graph edges per batch.

    The PlatoFormer ``graph_scattered_attention`` path materialises a tensor of
    shape ``[E, G*H, D_h]`` where ``E = sum_over_graphs(n_atoms_i^2)``. ``E``
    therefore grows *quadratically* in per-graph atom count, so a batch that
    fits the ``max_atoms`` budget can still OOM if it happens to be composed
    of a few large molecules. ``max_edges`` caps this quadratic term directly.
    """

    def __init__(
        self,
        dataset,
        max_batch_size,
        max_atoms=None,
        max_edges=None,
        shuffle=True,
        drop_last=False,
        seed=0,
        num_replicas=None,
        rank=None,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_atoms = max_atoms
        self.max_edges = max_edges
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._atom_count_cache = {}

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank}, expected in [0, {num_replicas - 1}]")

        self.num_replicas = num_replicas
        self.rank = rank

        dataset_length = len(self.dataset)
        if self.drop_last:
            self.num_samples = math.floor(dataset_length / self.num_replicas)
        else:
            self.num_samples = math.ceil(dataset_length / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _resolve_underlying_dataset_and_index(self, local_idx):
        dataset = self.dataset
        actual_idx = local_idx

        while isinstance(dataset, Subset):
            actual_idx = dataset.indices[actual_idx]
            dataset = dataset.dataset

        return dataset, int(actual_idx)

    def _get_num_atoms(self, local_idx):
        if local_idx in self._atom_count_cache:
            return self._atom_count_cache[local_idx]

        dataset, actual_idx = self._resolve_underlying_dataset_and_index(local_idx)

        if hasattr(dataset, "get_num_atoms"):
            num_atoms = int(dataset.get_num_atoms(actual_idx))
        elif hasattr(dataset, "get_atoms"):
            num_atoms = len(dataset.get_atoms(actual_idx))
        else:
            raise TypeError("Dataset must provide get_num_atoms(idx) or get_atoms(idx)")

        self._atom_count_cache[local_idx] = num_atoms
        return num_atoms

    def __len__(self):
        if self.max_atoms is None:
            return max(1, math.ceil(self.num_samples / self.max_batch_size))

        avg_atoms_per_graph = 50
        avg_graphs_per_batch = max(1, int(self.max_atoms / avg_atoms_per_graph))
        return max(1, math.ceil(self.num_samples / avg_graphs_per_batch))

    def _generate_indices(self):
        size = len(self.dataset)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(size, generator=generator).tolist()
        else:
            indices = list(range(size))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                full_repeats = (padding_size + len(indices) - 1) // len(indices)
                indices += (indices * full_repeats)[:padding_size]
        else:
            indices = indices[: self.total_size]

        if len(indices) < self.total_size:
            raise RuntimeError("Insufficient indices to cover replicas")

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return indices[: self.num_samples]

    def __iter__(self):
        indices = self._generate_indices()
        batch = []
        atom_total = 0
        edge_total = 0  # sum of n_atoms^2 across graphs in the batch

        for local_idx in indices:
            num_atoms = self._get_num_atoms(local_idx)
            num_edges = num_atoms * num_atoms

            if batch and (
                len(batch) >= self.max_batch_size
                or (self.max_atoms is not None and atom_total + num_atoms > self.max_atoms)
                or (self.max_edges is not None and edge_total + num_edges > self.max_edges)
            ):
                yield batch
                batch = []
                atom_total = 0
                edge_total = 0

            # A single sample that exceeds either cap: flush and either yield
            # it alone (if it still fits the edge cap by itself) or skip it.
            exceeds_atoms = self.max_atoms is not None and num_atoms > self.max_atoms
            exceeds_edges = self.max_edges is not None and num_edges > self.max_edges
            if exceeds_atoms or exceeds_edges:
                if batch:
                    yield batch
                    batch = []
                    atom_total = 0
                    edge_total = 0
                if exceeds_edges:
                    # Single molecule too large to fit the edge budget; drop it.
                    continue
                yield [local_idx]
                continue

            batch.append(local_idx)
            atom_total += num_atoms
            edge_total += num_edges

        if batch and (not self.drop_last or len(batch) == self.max_batch_size):
            yield batch


class OMol4mModule(L.LightningDataModule):
    """FairChem-compatible OMol 4M datamodule with graph-level charge/spin."""

    def __init__(
        self,
        data,
        batch_size,
        dynamic_batching=False,
        max_atoms_per_batch=None,
        max_atoms_per_batch_val=None,
        max_edges_per_batch=None,
        max_edges_per_batch_val=None,
        prefetch_factor=2,
        debug_subset=None,
        validation_mode="train_split",
        train_size=0.9,
        precompute_reference_energy=False,
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

    @staticmethod
    def _normalize_validation_mode(mode):
        if mode is None:
            return "train_split"

        mode_value = str(mode).strip().lower()
        if mode_value in {"train_split", "split_train", "from_train"}:
            return "train_split"
        if mode_value in {"heldout", "heldout_path", "from_heldout", "val_path"}:
            return "heldout"

        raise ValueError(
            "validation_mode must be one of: train_split, heldout"
        )

    def _build_dataset(self, data_path, keep_in_memory):
        return AseDBDatasetWithChargeSpin(
            config={
                "src": data_path,
                "a2g_args": dict(r_energy=True, r_forces=True),
                "keep_in_memory": keep_in_memory,
            },
            precompute_reference_energy=bool(getattr(self.hparams, "precompute_reference_energy", False)),
            reference_path=getattr(self.hparams, "reference_path", None),
        )

    @staticmethod
    def _train_val_split(dataset, train_size, seed):
        total_size = len(dataset)
        if total_size < 2:
            raise ValueError("Need at least 2 samples to split train/val")

        if train_size is None:
            raise ValueError("train_size cannot be None")

        if isinstance(train_size, float):
            if train_size <= 0:
                raise ValueError("train_size must be positive")
            if train_size <= 1:
                train_count = int(math.floor(train_size * total_size))
            elif train_size.is_integer():
                train_count = int(train_size)
            else:
                raise ValueError("train_size > 1 must be an integer-like value")
        else:
            train_count = int(train_size)
            if train_count <= 0:
                raise ValueError("train_size must be positive")

        train_count = max(1, min(train_count, total_size))
        val_count = total_size - train_count
        if val_count <= 0:
            val_count = 1
            train_count = total_size - 1

        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(total_size, generator=generator).tolist()

        train_indices = indices[:train_count]
        val_indices = indices[train_count : train_count + val_count]
        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    @staticmethod
    def _resolve_debug_subset_size(dataset_size, subset_spec):
        if isinstance(subset_spec, bool):
            raise ValueError("debug_subset must be a float percentage or integer count")

        if isinstance(subset_spec, float):
            if subset_spec <= 0:
                raise ValueError("debug_subset percentage must be > 0")
            if subset_spec <= 1:
                resolved = max(1, int(math.floor(dataset_size * subset_spec)))
            elif subset_spec.is_integer():
                resolved = int(subset_spec)
            else:
                raise ValueError("debug_subset > 1 must be an integer-like value")
        elif isinstance(subset_spec, int):
            if subset_spec <= 0:
                raise ValueError("debug_subset integer must be > 0")
            resolved = subset_spec
        else:
            try:
                parsed = float(subset_spec)
            except (TypeError, ValueError) as exc:
                raise ValueError("debug_subset must be a float percentage or integer count") from exc
            return OMol4mModule._resolve_debug_subset_size(dataset_size, parsed)

        return min(resolved, dataset_size)

    @staticmethod
    def _apply_debug_subset(dataset, subset_spec):
        subset_size = OMol4mModule._resolve_debug_subset_size(len(dataset), subset_spec)
        if subset_size >= len(dataset):
            return dataset
        return Subset(dataset, list(range(subset_size)))

    def setup(self, stage=None):
        # Lightning can call setup multiple times (fit -> test). Reuse existing
        # datasets to avoid reopening the same LMDB environment in-process.
        if self.datasets:
            return

        keep_in_memory = bool(getattr(self.hparams.data, "keep_in_memory", False))
        validation_mode = self._normalize_validation_mode(self.hparams.validation_mode)

        base_train_dataset = self._build_dataset(
            data_path=self.hparams.data.train_data_path,
            keep_in_memory=keep_in_memory,
        )

        if self.hparams.debug_subset is not None:
            base_train_dataset = self._apply_debug_subset(base_train_dataset, self.hparams.debug_subset)

        val_data_path = getattr(self.hparams.data, "val_data_path", None)
        test_data_path = getattr(self.hparams.data, "test_data_path", None)

        if validation_mode == "train_split":
            train_size = self.hparams.train_size
            if hasattr(self.hparams.data, "train_size") and self.hparams.data.train_size is not None:
                train_size = self.hparams.data.train_size

            train_dataset, val_dataset = self._train_val_split(
                dataset=base_train_dataset,
                train_size=train_size,
                seed=self.hparams.data.seed,
            )

            heldout_test_path = test_data_path or val_data_path
            if heldout_test_path is None:
                test_dataset = val_dataset
            else:
                test_dataset = self._build_dataset(
                    data_path=heldout_test_path,
                    keep_in_memory=keep_in_memory,
                )

            if self.hparams.debug_subset is not None:
                test_dataset = self._apply_debug_subset(test_dataset, self.hparams.debug_subset)

        elif validation_mode == "heldout":
            if val_data_path is None:
                raise ValueError(
                    "validation_mode=heldout requires data.val_data_path to be set"
                )

            train_dataset = base_train_dataset
            val_dataset = self._build_dataset(
                data_path=val_data_path,
                keep_in_memory=keep_in_memory,
            )

            if self.hparams.debug_subset is not None:
                val_dataset = self._apply_debug_subset(val_dataset, self.hparams.debug_subset)

            if test_data_path is not None and test_data_path != val_data_path:
                test_dataset = self._build_dataset(
                    data_path=test_data_path,
                    keep_in_memory=keep_in_memory,
                )
                if self.hparams.debug_subset is not None:
                    test_dataset = self._apply_debug_subset(test_dataset, self.hparams.debug_subset)
            else:
                test_dataset = val_dataset

        else:
            raise ValueError(f"Unsupported validation_mode: {validation_mode}")

        if self.hparams.debug_subset is not None:
            log.info(
                "debug_subset enabled: using %s train samples, %s val samples, %s test samples",
                len(train_dataset),
                len(val_dataset),
                len(test_dataset),
            )

        log.info("validation_mode=%s", validation_mode)

        self.datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        log.info(f"# Training: {len(self.datasets['train'])}")
        log.info(f"# Validation: {len(self.datasets['val'])}")
        log.info(f"# Test: {len(self.datasets['test'])}")

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

    @staticmethod
    def _is_rank_zero():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def _log_loader_batches(self, split_name, loader):
        if not self._is_rank_zero():
            return
        try:
            batch_count = len(loader)
        except TypeError:
            batch_count = "unknown"
        log.info("%s dataloader batches per epoch: %s", split_name, batch_count)

    def train_dataloader(self):
        loader = self._create_dataloader(
            self.datasets["train"],
            self.batch_size.train,
            shuffle=self.hparams.data.shuffle,
        )
        self._log_loader_batches("Train", loader)
        return loader

    def val_dataloader(self):
        loader = self._create_dataloader(
            self.datasets["val"],
            self.batch_size.val,
            shuffle=False,
        )
        self._log_loader_batches("Validation", loader)
        return loader

    def test_dataloader(self):
        test_batch_size = getattr(self.batch_size, "test", self.batch_size.val)
        loader = self._create_dataloader(
            self.datasets["test"],
            test_batch_size,
            shuffle=False,
        )
        self._log_loader_batches("Test", loader)
        return loader