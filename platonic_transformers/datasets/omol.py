import math
import os
import pickle
import threading
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm
from ase.neighborlist import neighbor_list
from mendeleev import element
from platonic_transformers.datasets.k_hot_encoding import KHOT_EMBEDDINGS  # This was from fairchem repo.

# a simple length cache for database
LENGTH_CACHE = {}
LENGTH_LOCK = threading.Lock()

class Batch:
    def __init__(self, data_dict):
        essential_fields = ['pos', 'x', 'energy', 'forces', 'batch', 'edge_index', 'edge_attr']
        metadata_fields = ['name', 'smiles', 'composition', 'idx']
        
        for key in essential_fields:
            if key in data_dict:
                setattr(self, key, data_dict[key])
        
        self._metadata = {k: data_dict[k] for k in metadata_fields if k in data_dict}
        
        for key in data_dict:
            if key not in essential_fields and key not in metadata_fields:
                setattr(self, key, data_dict[key])
    
    def __getitem__(self, key):
        """Allow dictionary-style access for both attributes and metadata."""
        if key in self._metadata:
            return self._metadata[key]
        return getattr(self, key)

    def __contains__(self, key):
        """Support `key in batch` (dict-style). Without this, Python falls
        back to int-indexed iteration through __getitem__, which raises
        TypeError because getattr() requires string keys. fairchem's
        eSCNMDBackbone does `"pbc" in data_dict`, so this is required for
        the eSEN baseline."""
        return key in self._metadata or hasattr(self, key)

    def get(self, key, default=None):
        """Dict-style .get() — fairchem's escn_md.py calls data_dict.get(...)."""
        if key in self._metadata:
            return self._metadata[key]
        return getattr(self, key, default)

    def __setitem__(self, key, value):
        """Allow dictionary-style assignment."""
        setattr(self, key, value)
    
    def keys(self):
        """Return all attribute names."""
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]
    
    def items(self):
        """Return key-value pairs like a dictionary."""
        return [(key, getattr(self, key)) for key in self.keys()]
    
    def to(self, device):
        """Move all tensor attributes to the specified device."""
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self
    
    def cuda(self):
        """Move all tensor attributes to CUDA."""
        return self.to('cuda')
    
    def cpu(self):
        """Move all tensor attributes to CPU."""
        return self.to('cpu')

    def __del__(self):
        """Explicitly break reference cycles when object is deleted."""
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                delattr(self, attr)
        if hasattr(self, '_metadata'):
            self._metadata.clear()

class OMolDataset(Dataset):
    def __init__(self, root='./data/omol',
                 split=None,
                 use_charges=False,
                 debug_subset=None,
                 seed=42,
                 energy_referencing=True,
                 edges=False,
                 edge_attr=False,
                 force_distance_method=True,
                 use_k_hot=False,
                 require_natoms_cache=False,
                 train_subdir='train_4M',
                 val_subdir='val'):
        
        # [Keep all the existing initialization code until dataset_path is set]
        
        self.energy_referencing = energy_referencing
        self.root = root
        self.split = split  
        self.use_charges = use_charges
        self.use_k_hot = use_k_hot  
        debug_subset = int(debug_subset) if debug_subset is not None else debug_subset 
        self.energy_coefficients = None 
        self.edges = edges  
        self.edge_attr = edge_attr
        self.force_distance_method = force_distance_method
        self.scale = 1.0
        self.shift = 0.0
       
        element_symbols = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa','U']
        
        self.dataset_info = {
            'name': 'omol',
            'atom_encoder': {symbol: i+1 for i, symbol in enumerate(element_symbols)},  # Start from 1, not 0
            'atom_decoder': element_symbols
        }
        
        # qcczbpfn-style data split: dedicated train_4M dir for training, and a
        # SEPARATE held-out val dir for validation (no internal 80/20 split).
        # Subdir names default to `train_4M` and `val` (matching the OMol25
        # release layout that qcczbpfn used), but can be overridden via the
        # constructor args (or yaml: `dataset.train_subdir`/`dataset.val_subdir`).
        if split == 'test' or split == 'val':
            self.dataset_path = os.path.join(self.root, val_subdir)
        else:
            self.dataset_path = os.path.join(self.root, train_subdir)

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"OMol dataset not found at {self.dataset_path}")

        # NEW CODE: Don't create AseDBDataset here - defer to workers
        # Instead, just get dataset length safely
        self._thread_local_dataset = None  # Each thread/process will have its own instance
        
        # Get dataset length safely (using cache)
        cache_key = self.dataset_path
        with LENGTH_LOCK:
            if cache_key in LENGTH_CACHE:
                dataset_length = LENGTH_CACHE[cache_key]
            else:
                # One-time initialization in the main process only for length
                # This instance will be discarded immediately after getting length
                tmp_dataset = AseDBDataset({"src": self.dataset_path})
                dataset_length = len(tmp_dataset)
                LENGTH_CACHE[cache_key] = dataset_length
                del tmp_dataset
        
        # Set indices based on length
        self.indices = list(range(dataset_length))

        # Load pre-computed per-sample atom counts from metadata.npz if present.
        # OMol25 ships a `metadata.npz` next to the .aselmdb shards with a
        # `natoms` array indexed the same as the dataset. Using this cache
        # turns `get_num_atoms(idx)` into an O(1) numpy lookup — required by
        # the DynamicAtomBatchSampler, which otherwise does one full LMDB
        # read per sample (drastically slow on a cold dataset over shared
        # GPFS — kills throughput entirely).
        #
        # Mirrors `AseDBDatasetWithChargeSpin._natoms_cache` in the private.
        # When `require_natoms_cache=True` (set by `get_omol_loaders` whenever
        # `dynamic_batching=True`) the cache MUST be present — we raise rather
        # than silently fall back to the slow path, because there's no
        # plausible reason to enable dynamic batching without the cache.
        self._natoms_cache = None
        metadata_path = os.path.join(self.dataset_path, "metadata.npz")
        cache_load_error = None
        if os.path.exists(metadata_path):
            try:
                with np.load(metadata_path, allow_pickle=False) as m:
                    if "natoms" in m.files:
                        natoms = np.asarray(m["natoms"], dtype=np.int64)
                        if len(natoms) == dataset_length:
                            self._natoms_cache = natoms
                            print(f"natoms cache loaded from {metadata_path} ({len(natoms)} samples)")
                        else:
                            cache_load_error = (
                                f"metadata.npz natoms length {len(natoms)} != dataset "
                                f"len {dataset_length} at {metadata_path}")
                    else:
                        cache_load_error = f"metadata.npz at {metadata_path} has no 'natoms' key"
            except Exception as exc:
                cache_load_error = f"failed to load metadata.npz at {metadata_path}: {exc}"
        else:
            cache_load_error = f"metadata.npz not found at {metadata_path}"

        if require_natoms_cache and self._natoms_cache is None:
            raise FileNotFoundError(
                f"natoms cache is required for dynamic batching but is unavailable: "
                f"{cache_load_error}.\n"
                f"OMol25 ships a `metadata.npz` next to the .aselmdb shards. If your "
                f"data came from the standard fairchem distribution it should already "
                f"be there. For custom shards or symlinked directories, run:\n"
                f"    python scripts/build_omol_natoms_cache.py {self.dataset_path}\n"
                f"to generate it (one-time, ~minutes). Or set "
                f"`training.dynamic_batching=false` in the yaml."
            )

        if debug_subset is not None:
            debug_subset = int(debug_subset)
            self.indices = self.indices[:debug_subset]

        self.split = split
        # No internal 80/20 split — train uses ALL of neutral_train (the full
        # 4M subset on Snellius) and val uses ALL of neutral_val. Matches the
        # qcczbpfn data wiring exactly.

    def _init_ase_dataset(self):
        """Safely initialize dataset in the worker process when needed"""
        if self._thread_local_dataset is None:
            self._thread_local_dataset = AseDBDataset({"src": self.dataset_path})
        return self._thread_local_dataset

    def get_num_atoms(self, idx):
        """Atom count for the molecule at index ``idx`` (consulted by the
        DynamicAtomBatchSampler to pack batches by atom budget). O(1) via the
        precomputed natoms cache when ``metadata.npz`` is present."""
        if self._natoms_cache is not None:
            return int(self._natoms_cache[int(idx)])
        ds = self._init_ase_dataset()
        if hasattr(ds, "get_num_atoms"):
            return int(ds.get_num_atoms(idx))
        if hasattr(ds, "get_atoms"):
            return int(len(ds.get_atoms(idx)))
        # Fallback: pull the row and count atomic_numbers.
        item = ds[idx]
        if hasattr(item, "atomic_numbers"):
            return int(len(item.atomic_numbers))
        if hasattr(item, "pos"):
            return int(len(item.pos))
        raise RuntimeError(f"Cannot determine num_atoms for idx={idx}")

    def set_scale_shift(self, scale=1.0, shift=0.0):
        """Set scale and shift for energy normalization."""
        self.scale = scale
        self.shift = shift
        print(f"Scale set to {self.scale}, Shift set to {self.shift} for {self.split} loader.")

    def apply_split(self):
        """Apply train/val split to neutral_train data using simple indexing."""
        total_size = len(self.indices)
        train_size = int(0.8 * total_size)  # Use 80% for train, 20% for val
        
        # No shuffling or permutation - use simple slicing
        if self.split == 'train':
            self.indices = self.indices[:train_size]
        elif self.split == 'val':
            self.indices = self.indices[train_size:]
        else:
            raise ValueError("Split must be 'train' or 'val' for internal splitting.")

    def create_edges(self, atoms, positions, edge_attr=False, force_distance_method=True, cutoff=6.0, max_neighbors=30):
        """Create edges using ASE neighbor list or distance-based method for proper connectivity."""
        
        num_atoms = len(atoms)
        edge_indices = []
        edge_attrs = []
        
        if force_distance_method:
            # Distance-based method with optional max_neighbors limit
            for i in range(num_atoms):
                distances_and_indices = []
                
                # Calculate distances to all other atoms
                for j in range(num_atoms):
                    if i != j:
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < cutoff and  dist > 1e-5:
                            distances_and_indices.append((dist, j))
                            
                # Sort by distance and limit to max_neighbors if specified
                distances_and_indices.sort(key=lambda x: x[0])
                if max_neighbors is not None:
                    distances_and_indices = distances_and_indices[:max_neighbors]
                
                # Add edges for this atom
                for dist, j in distances_and_indices:
                    edge_indices.append((i, j))
                    
                    if edge_attr:
                        # Simple edge attributes - default single bond
                        bond_type = [1.0, 0.0, 0.0, 0.0]
                        edge_attrs.append(bond_type)
        else:
            # Use ASE neighbor list to find connections 
            try:
                # Get neighbor pairs within cutoff distance
                i, j = neighbor_list('ij', atoms, cutoff)
                
                # Convert to edge list format
                edge_indices = list(zip(i, j))
                
                # Create edge attributes based on bond estimation
                if edge_attr:
                    for idx_i, idx_j in edge_indices:
                        dist = np.linalg.norm(positions[idx_i] - positions[idx_j])
                        bond_type = self.estimate_bond_type(atoms, idx_i, idx_j, dist)
                        edge_attrs.append(bond_type)

            except Exception as e:
                # Fallback to distance-based method if neighbor_list fails
                print(f"Warning: neighbor_list failed, falling back to distance method: {e}")
                return self.create_edges(atoms, positions, edge_attr=edge_attr, 
                                   force_distance_method=True, cutoff=cutoff, 
                                   max_neighbors=max_neighbors)
        
        if len(edge_indices) == 0:
            # No edges found, create empty arrays
            edge_index = np.zeros((2, 0), dtype=int)
            if edge_attr:
                edge_attr = np.zeros((0, 4), dtype=np.float32)
            else:
                edge_attr = None
        else:
            edge_index = np.array(edge_indices, dtype=int).T
            
            if edge_attr and edge_attrs:
                edge_attr = np.array(edge_attrs, dtype=np.float32)
                
                # Sort edges for consistency (like in QM9)
                sort_indices = np.lexsort((edge_index[0, :], edge_index[1, :]))
                edge_index = edge_index[:, sort_indices]
                edge_attr = edge_attr[sort_indices]
            else:
                edge_attr = None
                # Still sort edges for consistency
                sort_indices = np.lexsort((edge_index[0, :], edge_index[1, :]))
                edge_index = edge_index[:, sort_indices]
        
        return edge_index, edge_attr

    def estimate_bond_type(self, atoms, i, j, distance):
        """Estimate bond type based on atomic symbols and distance."""
        symbols = atoms.get_chemical_symbols()
        symbol_i, symbol_j = symbols[i], symbols[j]
        
        # Typical single bond distances (Angstrom)
        single_bond_distances = {
            ('C', 'C'): 1.54, ('C', 'H'): 1.09, ('C', 'N'): 1.47, ('C', 'O'): 1.43,
            ('N', 'N'): 1.45, ('N', 'H'): 1.01, ('N', 'O'): 1.40,
            ('O', 'O'): 1.48, ('O', 'H'): 0.96, ('H', 'H'): 0.74,
            ('C', 'S'): 1.81, ('S', 'S'): 2.05, ('S', 'H'): 1.34,
            ('C', 'Cl'): 1.77, ('C', 'F'): 1.35, ('C', 'Br'): 1.94
        }
        
        # Get expected single bond distance
        pair = tuple(sorted([symbol_i, symbol_j]))
        expected_single = single_bond_distances.get(pair, 1.8)  # Default 1.8 Å
        
        # Bond type estimation based on distance ratio
        ratio = distance / expected_single
        
        if ratio < 0.85:  # Much shorter than single bond
            return [0.0, 0.0, 1.0, 0.0]  # Triple bond
        elif ratio < 0.95:  # Shorter than single bond
            return [0.0, 1.0, 0.0, 0.0]  # Double bond
        elif ratio < 1.15:  # Around single bond length
            return [1.0, 0.0, 0.0, 0.0]  # Single bond
        else:  # Could be aromatic or weak bond
            return [0.0, 0.0, 0.0, 1.0]  # Aromatic/other

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ase_dataset = self._init_ase_dataset()
        
        actual_idx = self.indices[idx]
        atoms = ase_dataset.get_atoms(actual_idx)
        positions = atoms.get_positions().astype(np.float32)
        atomic_numbers = atoms.get_atomic_numbers()
        num_atoms = len(atoms)
        
        composition = atoms.get_chemical_formula(mode='hill')  
        
        if self.use_k_hot:
            # Use k-hot encoding for atom types
            k_hot_features = []
            for j, atomic_num in enumerate(atomic_numbers):
                if atomic_num in KHOT_EMBEDDINGS:
                    k_hot_features.append(KHOT_EMBEDDINGS[atomic_num])
                else:
                    symbol = atoms.get_chemical_symbols()[j]
                    print(f"Warning: Unknown element {symbol} (atomic number {atomic_num}) found in molecule {actual_idx}")
                    # Use all zeros for unknown elements
                    k_hot_features.append([0.0] * len(KHOT_EMBEDDINGS[1]))  # KHOT_EMBEDDINGS has 92-dimensional vectors
            
            # Convert to numpy array
            x = np.array(k_hot_features, dtype=np.float32)
        else:
            # Use one-hot encoding for atom types
            x = np.zeros((num_atoms, len(self.dataset_info['atom_encoder'])), dtype=np.float32)
            for j, atomic_num in enumerate(atomic_numbers):
                symbol = atoms.get_chemical_symbols()[j]
                if symbol in self.dataset_info['atom_encoder']:
                    # Use 1-based indexing (subtract 1 to convert to 0-based for array indexing)
                    x[j, self.dataset_info['atom_encoder'][symbol] - 1] = 1
                else:
                   
                    print(f"Warning: Unknown element {symbol} found in molecule {actual_idx}")
                    # Use all zeros for unknown elements (x[j] is already initialized to zeros)
                    pass
        
        # Get charges if available
        charges = np.zeros(num_atoms, dtype=np.float32)
        try:
            initial_charges = atoms.get_initial_charges()
            if initial_charges is not None:
                charges = initial_charges.astype(np.float32)
        except:
            pass
        
        # Get energy if available
        try:
            energy = np.array([atoms.get_potential_energy()], dtype=np.float32)
            if  self.energy_referencing and self.energy_coefficients is not None:
                energy[0] = normalize_energy(atoms, energy[0], self.energy_coefficients)
                # ipdb.set_trace()  # Debugging breakpoint
        except Exception as e:
            raise ValueError(f"Energy not found in molecule {actual_idx}: {str(e)}")
        
        try:
            forces = atoms.get_forces().astype(np.float32)
        except Exception as e:
            raise ValueError(f"Forces not found in molecule {actual_idx}: {str(e)}")
      
        # Get additional properties
        name = atoms.info.get('name', f'mol_{actual_idx}') if hasattr(atoms, 'info') else f'mol_{actual_idx}'
        smiles = atoms.info.get('smiles', '') if hasattr(atoms, 'info') else ''

        # Per-molecule charge/spin conditioning (qcczbpfn recipe). Stored under
        # atoms.info by the OMol25 distribution. Distinct from per-atom partial
        # charges in `atoms.get_initial_charges()` (which we also read above).
        charge = int(atoms.info.get('charge', 0)) if hasattr(atoms, 'info') else 0
        spin = int(atoms.info.get('spin', 0)) if hasattr(atoms, 'info') else 0
        
        # Convert to tensors
        x = torch.from_numpy(x)
        energy = torch.from_numpy(energy)
        forces = torch.from_numpy(forces)
        pos = torch.from_numpy(positions)
        
        if self.edges:
            edge_index, edge_attr = self.create_edges(atoms, positions, self.edge_attr,self.force_distance_method)
            edge_attr = torch.from_numpy(edge_attr) if edge_attr else None
            edge_index = torch.from_numpy(edge_index)
        else:
            edge_index = None
            edge_attr = None
        
        if self.use_charges:
            charges_tensor = torch.from_numpy(charges).unsqueeze(-1)
            x = torch.cat([x, charges_tensor], dim=-1)
        
        item = {
            'pos': pos,
            'x': x,
            'energy': energy,
            'forces': forces,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'name': name,
            'smiles': smiles,
            'composition': composition,
            'idx': actual_idx,
            'num_atoms': num_atoms,
            'charges': torch.from_numpy(charges),
            'atomic_numbers': torch.from_numpy(atomic_numbers),
            'charge': torch.tensor([charge], dtype=torch.long),
            'spin': torch.tensor([spin], dtype=torch.long),
        }
        
        return item

    def _init_ase_dataset(self):
        """Safely initialize dataset in the worker process when needed"""
        if self._thread_local_dataset is None:
            self._thread_local_dataset = AseDBDataset({"src": self.dataset_path})
        return self._thread_local_dataset

def collate_fn(batch):
    """Collate function for batching OMol data."""
    pos, x, energy, forces, batch_idx = [], [], [], [], []
    names, smiles_list, compositions, indices, num_atoms_list, charges_batch, atomic_numbers_batch = [], [], [], [], [], [], []
    charge_batch, spin_batch = [], []
    cum_nodes = 0
    
    # Check if edges are being used at all
    has_edges = batch[0]['edge_index'] is not None
    edge_index_batch, edge_attr_batch = ([], []) if has_edges else (None, None)
    
    for i, item in enumerate(batch):
        num_nodes = item['x'].shape[0]
        pos.append(item['pos'])
        x.append(item['x'])
        energy.append(item['energy'])
        forces.append(item['forces'])
        batch_idx.extend([i] * num_nodes)
        
        # Only process edges if they exist
        if has_edges:
            # Shift edge indices by cumulative node count
            edge_index = item['edge_index'].clone() + cum_nodes if item['edge_index'] is not None else None
            edge_index_batch.append(edge_index)
            edge_attr_batch.append(item['edge_attr'])
        
        # Collect metadata
        names.append(item['name'])
        smiles_list.append(item['smiles'])
        compositions.append(item['composition'])
        indices.append(item['idx'])
        num_atoms_list.append(item['num_atoms'])
        charges_batch.append(item['charges'])
        atomic_numbers_batch.append(item['atomic_numbers'])
        charge_batch.append(item['charge'])
        spin_batch.append(item['spin'])

        cum_nodes += num_nodes
    
    # Combine tensors
    pos = torch.cat(pos, dim=0)
    x = torch.cat(x, dim=0)
    energy = torch.stack(energy, dim=0).squeeze(-1)
    if energy.dim() == 0:
        energy = energy.unsqueeze(0)
    forces = torch.cat(forces, dim=0)
    batch_idx = torch.tensor(batch_idx, dtype=torch.long)
    
    # Process edges only if they exist
    if has_edges and all(e is not None for e in edge_index_batch):
        edge_index = torch.cat(edge_index_batch, dim=1)
        edge_attr = (torch.cat([attr for attr in edge_attr_batch if attr is not None], dim=0) 
                    if any(attr is not None for attr in edge_attr_batch) else None)
    else:
        edge_index = None
        edge_attr = None
    
    charges = torch.cat(charges_batch, dim=0)
    atomic_numbers = torch.cat(atomic_numbers_batch, dim=0)
    num_of_atoms = torch.tensor(num_atoms_list, dtype=torch.long)
    charge = torch.cat(charge_batch, dim=0)
    spin = torch.cat(spin_batch, dim=0)

    batch_dict = {
        'pos': pos, 'x': x, 'energy': energy, 'forces': forces, 'batch': batch_idx,
        'edge_index': edge_index, 'edge_attr': edge_attr, 'name': names, 'smiles': smiles_list,
        'composition': compositions, 'idx': indices, 'num_atoms': num_of_atoms,
        'charges': charges, 'atomic_numbers': atomic_numbers, 'cum_nodes': torch.tensor(cum_nodes),
        'charge': charge, 'spin': spin,
    }
    
    # Clean up to prevent memory leaks
    if has_edges:
        del edge_index_batch, edge_attr_batch
    del names, smiles_list, compositions, indices, num_atoms_list, charges_batch, atomic_numbers_batch
    
    return Batch(batch_dict)


def get_per_atom_energy_and_stat(dataset, coef_path=None, recalculate=False,include_hof=False):
    """
    Get energy per-atom energy for referencing plus get mean, std and average number of nodes, either loading from file or computing.
    
    Args:
        dataset: An OMolDataset instance
        coef_path: Path to save/load coefficients
        recalculate: Force recalculation even if file exists
        include_hof: Whether to include heat of formation in energy referencing
    Returns:
        dict: Per-element energy coefficients
    """
    # Try to load from file if path provided and not forcing recalculation

    coef_path = os.path.join(coef_path, f'per_atom_energy_hof_{str(include_hof)}.pkl') 
    
    if os.path.exists(coef_path) and not recalculate:
        print(f"Loading energy normalization coefficients from {coef_path}")
        with open(coef_path, 'rb') as f:
            stats = pickle.load(f)
            per_elm_energy = stats['per_elm_energy']
            dataset_stats = stats['dataset_stats']

            return per_elm_energy, dataset_stats
        
 
    return compute_per_atom_energy_and_stat(dataset, save_path=coef_path, include_hof=include_hof)

def compute_per_atom_energy_and_stat(dataset, save_path=None, include_hof=True, use_rmsd=True):
    """
    Compute per-element energy coefficients using linear regression.
    
    The formula used is: E_ref = E_DFT - Σ[E_i,DFT - ΔH_f,i]
    
    Args:
        dataset: An OMolDataset instance
        save_path: Path to save the coefficients (optional)
        include_hof: Whether to include heat of formation in energy referencing
        use_rmsd: Whether to use RMSD with Bessel's correction (True) or std (False) for scale
    
    Returns:
        tuple: (per_elm_energy, dataset_stats) where dataset_stats contains
               mean, std, and avg_num_nodes of normalized energies
    """
    
    print("Computing per-element energy for referencing...")
    num_elements = len(dataset.dataset_info['atom_decoder'])
    print(f"Using {num_elements} elements")
    
    # Get heat of formation values for all elements
    print("Getting heat of formation values from mendeleev...")
    heat_of_formation = {}
    for elem_symbol in dataset.dataset_info['atom_decoder']:
        if include_hof:
            try:
                elem = element(elem_symbol)
                hof = elem.heat_of_formation if elem.heat_of_formation is not None else 0.0
                heat_of_formation[elem_symbol] = hof * 0.0103642  # Convert kJ/mol to eV
                # print(f"{elem_symbol}: ΔH_f = {elem.heat_of_formation} eV")
            except Exception as e:
                print(f"Warning: Could not get heat of formation for {elem_symbol}: {e}")
                heat_of_formation[elem_symbol] = 0.0
        else:
            # If not including HOF, set to zero
            heat_of_formation[elem_symbol] = 0.0
    
    K_matrix = []  # Element counts for each molecule
    E_dft = []     # DFT energy for each molecule
    num_atoms_list = []  # Number of atoms per molecule for avg_num_nodes
    
    # Loop through dataset to build K matrix and E_dft vector
    print("Building K matrix and E_DFT vector...")
    for idx in tqdm(dataset.indices):
        atoms = dataset.ase_dataset.get_atoms(idx)
        energy_dft = atoms.get_potential_energy()
        
        # Count elements in this molecule
        element_counts = {}
        for symbol in atoms.get_chemical_symbols():
            if symbol in element_counts:
                element_counts[symbol] += 1
            else:
                element_counts[symbol] = 1
        
        # Create a fixed-size row vector for this molecule's element counts
        K_row = np.zeros(num_elements, dtype=np.float32)
        for i, element_symbol in enumerate(dataset.dataset_info['atom_decoder']):
            K_row[i] = element_counts.get(element_symbol, 0)
       
        K_matrix.append(K_row)
        E_dft.append(energy_dft)
        num_atoms_list.append(len(atoms))
    

    K = np.array(K_matrix)
    E_dft = np.array(E_dft)
    num_atoms_array = np.array(num_atoms_list)
    
    print(f"K matrix shape: {K.shape}, E_dft shape: {E_dft.shape}")
    print(f"Energy range: min={E_dft.min():.4f}, max={E_dft.max():.4f}, mean={E_dft.mean():.4f}")
    
    # Reduce the composition matrix to only features that are non-zero to improve rank
    mask = K.sum(axis=0) != 0.0
    reduced_K = K[:, mask]
    print(f"Reduced K matrix shape: {reduced_K.shape} (filtered out {K.shape[1] - reduced_K.shape[1]} zero columns)")
    
    # Replace sklearn with numpy.linalg.lstsq
    print("Solving linear regression K*P = E_DFT using numpy.linalg.lstsq...")
    coeffs_reduced, residuals, rank, s = np.linalg.lstsq(reduced_K, E_dft, rcond=None)
    
    # Extract isolated atomic energies E_i,DFT
    E_isolated = {}
    coeffs = np.zeros(K.shape[1])
    coeffs[mask] = coeffs_reduced
    
    for i, element_symbol in enumerate(dataset.dataset_info['atom_decoder']):
        E_isolated[element_symbol] = coeffs[i]
    
    # Calculate predictions and R² score manually
    E_pred = K @ coeffs
    ss_total = np.sum((E_dft - np.mean(E_dft))**2)
    ss_residual = np.sum((E_dft - E_pred)**2)
    r2_score = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0
    
    print(f"\nLinear regression R² score: {r2_score:.6f}")
    print(f"Mean absolute error: {np.mean(np.abs(E_pred - E_dft)):.4f} eV")
    
    # Now calculate the per-atom energy contributions (E_i,DFT - ΔH_f,i) for normalization
    per_elm_energy = {}
    for element_symbol in dataset.dataset_info['atom_decoder']:
        per_elm_energy[element_symbol] = E_isolated[element_symbol] - heat_of_formation[element_symbol]
    
    print("\nLearned per-element coefficients:")
    for element_symbol in dataset.dataset_info['atom_decoder'][:10]:
        if element_symbol in per_elm_energy:
            e_isolated = E_isolated[element_symbol]
            hof_val = heat_of_formation[element_symbol]
            p_val = per_elm_energy[element_symbol]
            print(f"{element_symbol}: E_DFT={e_isolated:.4f} eV, ΔH_f={hof_val:.4f} eV, P=(E_DFT-ΔH_f)={p_val:.4f} eV")
        
    E_norm = []
    for idx, (k_row, e_orig) in enumerate(zip(K, E_dft)):
        sum_contributions = 0
        for i, element_symbol in enumerate(dataset.dataset_info['atom_decoder']):
            if k_row[i] > 0: 
                sum_contributions += k_row[i] * per_elm_energy[element_symbol]
        
        e_norm = e_orig - sum_contributions
        E_norm.append(e_norm)
    
    E_norm = np.array(E_norm)
    print(f"\nNormalized energy range: min={E_norm.min():.4f}, max={E_norm.max():.4f}, mean={E_norm.mean():.4f}")
    
    # Compute dataset statistics from normalized energies
    mean_energy = float(np.mean(E_norm))
    
    if use_rmsd:
        # Use RMSD with Bessel's correction when mean != 0
        rmsd_correction = 0 if mean_energy == 0.0 else 1
        scale = float(np.sqrt(np.sum((E_norm - mean_energy)**2) / max(len(E_norm) - rmsd_correction, 1)))
        scale_method = "RMSD"
    else:
        # Use standard deviation (old method)
        scale = float(np.std(E_norm))
        scale_method = "std"
    
    dataset_stats = {
        'shift': mean_energy,
        'scale': scale,
        'avg_num_nodes': float(np.mean(num_atoms_array))
    }
    
    print(f"Dataset statistics - Mean: {dataset_stats['shift']:.4f}, Scale ({scale_method}): {dataset_stats['scale']:.4f}, Avg nodes: {dataset_stats['avg_num_nodes']:.1f}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        coefficients_data = {
            'per_elm_energy': per_elm_energy,
            'E_isolated': E_isolated,
            'heat_of_formation': heat_of_formation,
            'r2_score': r2_score,
            'mean_abs_error': np.mean(np.abs(E_pred - E_dft)),
            'include_hof': include_hof,
            'dataset_stats': dataset_stats
        }
        with open(save_path, 'wb') as f:
            pickle.dump(coefficients_data, f)
        print(f"Saved energy per-atom and dataset statistics to {save_path}")
    
    print("Energy per-atom and stats computed successfully")
    return per_elm_energy, dataset_stats

def normalize_energy(atoms, energy_dft, hof_sub_per_atom_energy):
    """
    Normalize energy using pre-computed coefficients following HOF reference.
    
    The formula used is: E_ref = E_DFT - Σ[E_i,DFT - ΔH_f,i]
    
    Args:
        atoms: ASE Atoms object
        energy_dft: Original DFT energy
        hof_sub_per_atom_energy: Dictionary of per-element energy coefficients
                           These contain (E_i,DFT - ΔH_f,i) for each element
    
    Returns:
        float: Normalized energy value (HOF referenced)
    """
    # Get element counts for this molecule (K_i row)
    element_counts = {}
    for symbol in atoms.get_chemical_symbols():
        if symbol in element_counts:
            element_counts[symbol] += 1
        else:
            element_counts[symbol] = 1
    
    # Calculate sum of per-element contributions: Σ[count_i * (E_i,DFT - ΔH_f,i)]
    # Note: energy_coefficients contains the (E_i,DFT - ΔH_f,i) values
    sum_atomic_contributions = 0.0
    for element_symbol, count in element_counts.items():
        if element_symbol in hof_sub_per_atom_energy:
            contribution = count * hof_sub_per_atom_energy[element_symbol]
            sum_atomic_contributions += contribution
        else:
            print(f"Warning: No coefficient found for element {element_symbol}")
    
    # Apply HOF reference: E_ref = E_DFT - Σ[count_i * (E_i,DFT - ΔH_f,i)]
    normalized_energy = energy_dft - sum_atomic_contributions
    
    return normalized_energy

def compute_stats(dataset, save_path=None, use_rmsd=True):
    """
    Compute simple mean and std without per-element referencing.
    
    Args:
        dataset: An OMolDataset instance
        save_path: Path to save the statistics (optional)
        use_rmsd: Whether to use RMSD with Bessel's correction (True) or std (False) for scale
    
    Returns:
        tuple: (None, dataset_stats) where dataset_stats contains
               mean, std, and avg_num_nodes of raw energies
    """
    print("Computing simple dataset statistics (no per-element referencing)...")
    
    E_dft = []
    num_atoms_list = []
    
    print("Collecting energies and atom counts...")
    for idx in tqdm(dataset.indices):
        atoms = dataset.ase_dataset.get_atoms(idx)
        energy_dft = atoms.get_potential_energy()
        E_dft.append(energy_dft)
        num_atoms_list.append(len(atoms))
    
    E_dft = np.array(E_dft)
    num_atoms_array = np.array(num_atoms_list)
    
    print(f"Raw energy range: min={E_dft.min():.4f}, max={E_dft.max():.4f}, mean={E_dft.mean():.4f}")
    
    # Compute dataset statistics from raw energies
    mean_energy = float(np.mean(E_dft))
    
    if use_rmsd:
        # Use RMSD with Bessel's correction when mean != 0
        rmsd_correction = 0 if mean_energy == 0.0 else 1
        scale = float(np.sqrt(np.sum((E_dft - mean_energy)**2) / max(len(E_dft) - rmsd_correction, 1)))
        scale_method = "RMSD"
    else:
        # Use standard deviation (old method)
        scale = float(np.std(E_dft))
        scale_method = "std"
    
    dataset_stats = {
        'shift': mean_energy,
        'scale': scale,
        'avg_num_nodes': float(np.mean(num_atoms_array))
    }
    
    print(f"Dataset statistics - Mean: {dataset_stats['shift']:.4f}, Scale ({scale_method}): {dataset_stats['scale']:.4f}, Avg nodes: {dataset_stats['avg_num_nodes']:.1f}")
    
   
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stats_data = {
            'per_elm_energy': None,  
            'dataset_stats': dataset_stats
        }
        with open(save_path, 'wb') as f:
            pickle.dump(stats_data, f)
        print(f"Saved simple dataset statistics to {save_path}")
    
    return None, dataset_stats


class DynamicAtomBatchSamplerForAseDB(Sampler):
    """Batch sampler that caps graph count, total atoms, and (optionally) total
    within-graph edges per batch.

    Verbatim port of ``DynamicAtomBatchSamplerForAseDB`` from the private
    ``src/data/omol_4m_module.py`` — same algorithm qcczbpfn used to pack
    batches up to ``max_atoms_per_batch=3000`` / ``max_edges_per_batch=600000``.

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
            exceeds_atoms = self.max_atoms is not None and num_atoms > self.max_atoms
            exceeds_edges = self.max_edges is not None and num_edges > self.max_edges
            if exceeds_atoms or exceeds_edges:
                if batch:
                    yield batch
                    batch = []
                    atom_total = 0
                    edge_total = 0
                if exceeds_edges:
                    continue
                yield [local_idx]
                continue
            batch.append(local_idx)
            atom_total += num_atoms
            edge_total += num_edges
        if batch and (not self.drop_last or len(batch) == self.max_batch_size):
            yield batch


def get_omol_loaders(root='/ssdstore/omol/', batch_size=32, num_workers=4,
                     use_charges=False, seed=42, debug_subset=None, include_hof=False,
                     referencing=True, scale_shift=False, recalculate=False, use_k_hot=False,
                     edge=False, edge_attr=False, force_distance_method=True,
                     dynamic_batching=False, max_atoms_per_batch=None,
                     max_atoms_per_batch_val=None, max_edges_per_batch=None,
                     max_edges_per_batch_val=None,
                     train_subdir='train_4M', val_subdir='val'):
    """
    Create DataLoaders for train, validation, and test splits of OMol dataset.
    
    Args:
        root (str): Path to the dataset root directory
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of workers for DataLoaders
        use_charges (bool): Whether to include charge information
        seed (int): Random seed for reproducible splits
        debug_subset (int, optional): Use only first N samples for debugging
        include_hof (bool): Include heat of formation in energy referencing
        referencing (bool): Whether to use per-atom referencing for the target property
        scale_shift (bool): Whether to use scale and shift normalization for the target property
        use_k_hot (bool): Whether to use k-hot encoding instead of one-hot encoding
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, energy_coefficients, dataset_stats)
    """
 
    # Create datasets for each split. When dynamic_batching is enabled the
    # DynamicAtomBatchSampler needs O(1) atom-count lookups via metadata.npz
    # — pass `require_natoms_cache=True` so OMolDataset raises a clear error
    # instead of silently falling back to slow per-sample LMDB reads.
    _common = dict(
        use_charges=use_charges, debug_subset=debug_subset, seed=seed,
        energy_referencing=referencing, use_k_hot=use_k_hot,
        edges=edge, edge_attr=edge_attr,
        force_distance_method=force_distance_method,
        require_natoms_cache=bool(dynamic_batching),
        train_subdir=train_subdir, val_subdir=val_subdir,
    )
    train_dataset = OMolDataset(root=root, split='train', **_common)
    val_dataset = OMolDataset(root=root, split='val', **_common)
    test_dataset = OMolDataset(root=root, split='test', **_common)

    energy_coefficients = None
    dataset_stats = None
    
    if not scale_shift:
        print("Scale/shift normalization disabled - using default values")
        dataset_stats = {
            'shift': 0.0,
            'scale': 1.0,
            'avg_num_nodes': 1.0
        }
        
        if referencing:
            print("Computing per-element coefficients (without using dataset stats for scale/shift)")
            energy_coefficients, _ = get_per_atom_energy_and_stat(
                dataset=train_dataset, 
                coef_path=root, 
                include_hof=include_hof,
                recalculate=recalculate
            )
            
            train_dataset.energy_coefficients = energy_coefficients
            val_dataset.energy_coefficients = energy_coefficients
            test_dataset.energy_coefficients = energy_coefficients
    elif referencing:
        print("Per-atom referencing enabled - computing per-element energy")
        energy_coefficients, dataset_stats = get_per_atom_energy_and_stat(
            dataset=train_dataset, 
            coef_path=root, 
            include_hof=include_hof,
            recalculate=recalculate
        )

        train_dataset.energy_coefficients = energy_coefficients
        val_dataset.energy_coefficients = energy_coefficients
        test_dataset.energy_coefficients = energy_coefficients
    else:
        print("Per-atom referencing disabled - computing simple statistics from raw energies")
        simple_stats_path = os.path.join(root, 'simple_stats.pkl')
        if os.path.exists(simple_stats_path) and not recalculate:
            print(f"Loading simple statistics from {simple_stats_path}")
            with open(simple_stats_path, 'rb') as f:
                stats_data = pickle.load(f)
                energy_coefficients = stats_data.get('per_elm_energy', None)
                dataset_stats = stats_data['dataset_stats']
        else:
            energy_coefficients, dataset_stats = compute_stats(
                dataset=train_dataset, 
                save_path=os.path.join(root, 'simple_stats.pkl')
            )

    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.set_scale_shift(scale=dataset_stats['scale'], shift=dataset_stats['shift'])
  

    # When dynamic_batching=True we replace the fixed-size batching with the
    # private's DynamicAtomBatchSamplerForAseDB, packing each batch up to a
    # ``max_atoms_per_batch`` (and optionally ``max_edges_per_batch``) budget
    # rather than a fixed graph count. ``batch_size`` then becomes the cap on
    # graphs per batch (qcczbpfn used 16 for train, 16 for val).
    common = dict(
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        multiprocessing_context='spawn',
        prefetch_factor=4 if num_workers > 0 else None,  # qcczbpfn parity
    )
    if dynamic_batching:
        # qcczbpfn parity: with dynamic batching on, the per-batch graph cap is
        # effectively disabled — batches are bounded only by max_atoms /
        # max_edges. Hipster's `_create_dataloader` passes max_batch_size=999999
        # in this branch; the `batch_size` yaml field is for the static-batching
        # fallback only. Passing batch_size as the graph cap here would
        # underfill batches whenever mean(atoms/molecule) × batch_size <
        # max_atoms_per_batch (the case for OMol25-4M: 55 × 64 = 3520 ≪ 12000).
        _GRAPH_CAP = 999999
        train_sampler = DynamicAtomBatchSamplerForAseDB(
            train_dataset, max_batch_size=_GRAPH_CAP,
            max_atoms=max_atoms_per_batch, max_edges=max_edges_per_batch,
            shuffle=True, drop_last=False, seed=seed,
        )
        val_sampler = DynamicAtomBatchSamplerForAseDB(
            val_dataset, max_batch_size=_GRAPH_CAP,
            max_atoms=max_atoms_per_batch_val or max_atoms_per_batch,
            max_edges=max_edges_per_batch_val or max_edges_per_batch,
            shuffle=False, drop_last=False, seed=seed,
        )
        test_sampler = DynamicAtomBatchSamplerForAseDB(
            test_dataset, max_batch_size=_GRAPH_CAP,
            max_atoms=max_atoms_per_batch_val or max_atoms_per_batch,
            max_edges=max_edges_per_batch_val or max_edges_per_batch,
            shuffle=False, drop_last=False, seed=seed,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **common)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **common)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, **common)
        # Attach `set_epoch` to the loader itself so Lightning calls our custom
        # batch_sampler's set_epoch between epochs (Lightning only looks for
        # set_epoch on `loader.sampler` or on the loader directly; with
        # `batch_sampler=` and use_distributed_sampler=False the sampler.set_epoch
        # is otherwise never called → same shuffle every epoch, defeating
        # shuffling on DDP and 1-GPU alike). Mirrors hipster's
        # `_create_dataloader` in omol_4m_module.py.
        train_loader.set_epoch = train_sampler.set_epoch
        val_loader.set_epoch = val_sampler.set_epoch
        test_loader.set_epoch = test_sampler.set_epoch
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **common)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)
    
    if debug_subset is not None:
        print(f"Debug mode: Using subset of {debug_subset} samples")
    print(f"Created OMol DataLoaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples") 
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, energy_coefficients, dataset_stats

