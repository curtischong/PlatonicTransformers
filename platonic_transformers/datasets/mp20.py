from __future__ import annotations

import math
import os
import csv
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split

try:
    from pymatgen.core import Structure
except ImportError as exc:  # pragma: no cover - exercised only without optional dep
    Structure = None
    _PYMATGEN_IMPORT_ERROR = exc
else:
    _PYMATGEN_IMPORT_ERROR = None


_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
_ELEMENT_TO_Z = {symbol: idx + 1 for idx, symbol in enumerate(_ELEMENTS)}


@dataclass
class _Crystal:
    atomic_numbers: list[int]
    cart_coords: list[list[float]]
    cell: list[list[float]]
    volume: float


def _iter_cif_paths(root: Path):
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".cif"):
                yield Path(entry.path)


@dataclass
class CrystalBatch:
    pos: Tensor
    x: Tensor
    y: Tensor
    batch: Tensor
    cell: Tensor
    pbc: Tensor
    num_atoms: Tensor
    atomic_numbers: Tensor
    material_ids: list[str]
    volumes: Tensor

    def to(self, device: torch.device | str) -> "CrystalBatch":
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                setattr(self, name, value.to(device))
        return self


def _parse_cif_number(value: str) -> float:
    value = value.strip().strip("'\"")
    if "(" in value:
        value = value.split("(", 1)[0]
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def _cell_from_lengths_angles(
    a: float,
    b: float,
    c: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> list[list[float]]:
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)

    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * cos_gamma, b * sin_gamma, 0.0
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sq = max(c * c - cx * cx - cy * cy, 0.0)
    cz = math.sqrt(cz_sq)
    return [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]


def _det3(cell: list[list[float]]) -> float:
    a, b, c = cell
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _fast_parse_cif_text(cif: str, source: str = "<cif>") -> _Crystal:
    lines = cif.splitlines()
    scalars: dict[str, str] = {}
    atom_headers: list[str] | None = None
    atom_rows: list[list[str]] = []

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if stripped.startswith("_"):
            parts = shlex.split(stripped, comments=False, posix=True)
            if len(parts) >= 2:
                scalars[parts[0]] = parts[1]
            i += 1
            continue
        if stripped == "loop_":
            i += 1
            headers = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip())
                i += 1
            rows = []
            while i < len(lines):
                row = lines[i].strip()
                if (
                    not row
                    or row == "loop_"
                    or row.startswith("_")
                    or row.startswith("data_")
                ):
                    break
                rows.append(shlex.split(row, comments=False, posix=True))
                i += 1
            if "_atom_site_fract_x" in headers and "_atom_site_fract_y" in headers:
                atom_headers = headers
                atom_rows = rows
            continue
        i += 1

    if atom_headers is None:
        raise ValueError(f"No atom_site fractional-coordinate loop found in {source}")

    a = _parse_cif_number(scalars["_cell_length_a"])
    b = _parse_cif_number(scalars["_cell_length_b"])
    c = _parse_cif_number(scalars["_cell_length_c"])
    alpha = _parse_cif_number(scalars["_cell_angle_alpha"])
    beta = _parse_cif_number(scalars["_cell_angle_beta"])
    gamma = _parse_cif_number(scalars["_cell_angle_gamma"])
    cell = _cell_from_lengths_angles(a, b, c, alpha, beta, gamma)
    volume = abs(_det3(cell))

    col = {name: idx for idx, name in enumerate(atom_headers)}
    symbol_col = col.get("_atom_site_type_symbol", col.get("_atom_site_label"))
    if symbol_col is None:
        raise ValueError(f"No atom-site element column found in {source}")
    x_col = col["_atom_site_fract_x"]
    y_col = col["_atom_site_fract_y"]
    z_col = col["_atom_site_fract_z"]

    atomic_numbers = []
    cart_coords = []
    for row in atom_rows:
        symbol = "".join(ch for ch in row[symbol_col] if ch.isalpha())
        if symbol not in _ELEMENT_TO_Z:
            raise ValueError(f"Unknown element symbol {symbol!r} in {source}")
        frac = [
            _parse_cif_number(row[x_col]),
            _parse_cif_number(row[y_col]),
            _parse_cif_number(row[z_col]),
        ]
        cart = [
            frac[0] * cell[0][dim] + frac[1] * cell[1][dim] + frac[2] * cell[2][dim]
            for dim in range(3)
        ]
        atomic_numbers.append(_ELEMENT_TO_Z[symbol])
        cart_coords.append(cart)

    return _Crystal(atomic_numbers, cart_coords, cell, volume)


def _fast_read_cif(path: Path) -> _Crystal:
    return _fast_parse_cif_text(path.read_text(), source=str(path))


def _crystal_to_item(
    structure: _Crystal,
    *,
    material_id: str,
    target_value: float,
    max_atomic_number: int,
    properties: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    atomic_numbers = torch.tensor(structure.atomic_numbers, dtype=torch.long)
    num_atoms = len(atomic_numbers)

    x = torch.zeros(num_atoms, max_atomic_number, dtype=torch.float32)
    valid = (atomic_numbers >= 1) & (atomic_numbers <= max_atomic_number)
    if valid.any():
        atom_rows = torch.arange(num_atoms)[valid]
        x[atom_rows, atomic_numbers[valid] - 1] = 1.0

    item = {
        "pos": torch.tensor(structure.cart_coords, dtype=torch.float32),
        "x": x,
        "y": torch.tensor([target_value], dtype=torch.float32),
        "cell": torch.tensor(structure.cell, dtype=torch.float32),
        "pbc": torch.ones(3, dtype=torch.bool),
        "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
        "atomic_numbers": atomic_numbers,
        "material_id": material_id,
        "volume": torch.tensor(float(structure.volume), dtype=torch.float32),
    }
    if properties:
        item.update(properties)
    return item


class MP20CIFDataset(Dataset):
    """Small CIF-backed crystal dataset for periodic PlatonicTransformer checks.

    The public MP20 benchmark normally ships material properties alongside
    structures. Local scratch copies are often just CIF directories, so this
    loader provides geometry-derived regression targets for smoke training:
    ``log_volume_per_atom`` (default), ``volume_per_atom``, ``volume``, and
    ``num_atoms``. The default parser reads the listed CIF atom sites without
    symmetry expansion so smoke runs stay fast; pass ``parser="pymatgen"`` for
    full pymatgen parsing.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        limit: Optional[int] = None,
        max_atoms: Optional[int] = None,
        target: str = "log_volume_per_atom",
        max_atomic_number: int = 118,
        parser: str = "fast",
    ) -> None:
        if parser not in {"fast", "pymatgen"}:
            raise ValueError("parser must be 'fast' or 'pymatgen'")
        if parser == "pymatgen" and Structure is None:
            raise ImportError(
                "MP20CIFDataset requires pymatgen. Install with `pip install pymatgen`."
            ) from _PYMATGEN_IMPORT_ERROR

        self.root = Path(root)
        self.target = target
        self.max_atomic_number = max_atomic_number
        self.parser = parser
        self._structure_cache = {}
        if limit is not None:
            files_iter = _iter_cif_paths(self.root)
        else:
            files_iter = iter(sorted(self.root.glob("*.cif")))

        if max_atoms is not None:
            filtered = []
            for path in files_iter:
                try:
                    structure = self._read_structure(path)
                    if len(structure.atomic_numbers) <= max_atoms:
                        filtered.append(path)
                        self._structure_cache[path] = structure
                except Exception:
                    continue
                if limit is not None and len(filtered) >= int(limit):
                    break
            self.files = filtered
        else:
            if limit is None:
                self.files = list(files_iter)
            else:
                self.files = []
                for path in files_iter:
                    self.files.append(path)
                    if len(self.files) >= int(limit):
                        break
        if not self.files:
            raise FileNotFoundError(f"No CIF files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, object]:
        path = self.files[idx]
        structure = self._structure_cache.get(path)
        if structure is None:
            structure = self._read_structure(path)

        target_value = self._target_value(float(structure.volume), len(structure.atomic_numbers))
        return _crystal_to_item(
            structure,
            material_id=path.stem.replace("MP_", ""),
            target_value=target_value,
            max_atomic_number=self.max_atomic_number,
        )

    def _read_structure(self, path: Path) -> _Crystal:
        if self.parser == "fast":
            return _fast_read_cif(path)

        structure = Structure.from_file(str(path))
        return _Crystal(
            atomic_numbers=list(structure.atomic_numbers),
            cart_coords=structure.cart_coords.tolist(),
            cell=structure.lattice.matrix.tolist(),
            volume=float(structure.volume),
        )

    def _target_value(self, volume: float, num_atoms: int) -> float:
        if self.target == "volume":
            return volume
        if self.target == "volume_per_atom":
            return volume / num_atoms
        if self.target == "log_volume_per_atom":
            return math.log(volume / num_atoms)
        if self.target == "num_atoms":
            return float(num_atoms)
        raise ValueError(
            "target must be one of 'log_volume_per_atom', 'volume_per_atom', "
            f"'volume', or 'num_atoms', got {self.target!r}"
        )


class MP20CSVRegressionDataset(Dataset):
    """CSV-backed MP20 regression dataset with real MP benchmark labels."""

    def __init__(
        self,
        csv_path: str | os.PathLike,
        *,
        target: str = "formation_energy_per_atom",
        max_atomic_number: int = 118,
        limit: Optional[int] = None,
        preload: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.target = target
        self.max_atomic_number = max_atomic_number
        self.rows: list[dict[str, str]] = []
        self.targets: list[float] = []
        self._item_cache: list[Optional[dict[str, object]]] | None = [] if preload else None

        with open(self.csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            if target not in (reader.fieldnames or []):
                raise ValueError(
                    f"Target {target!r} not found in {self.csv_path}. "
                    f"Available columns: {reader.fieldnames}"
                )
            for row in reader:
                self.rows.append(row)
                self.targets.append(float(row[target]))
                if self._item_cache is not None:
                    self._item_cache.append(None)
                if limit is not None and len(self.rows) >= int(limit):
                    break

        if not self.rows:
            raise FileNotFoundError(f"No rows found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        if self._item_cache is not None and self._item_cache[idx] is not None:
            return self._item_cache[idx]

        row = self.rows[idx]
        material_id = row.get("material_id", f"row_{idx}")
        structure = _fast_parse_cif_text(row["cif"], source=f"{self.csv_path}:{material_id}")
        item = _crystal_to_item(
            structure,
            material_id=material_id,
            target_value=float(row[self.target]),
            max_atomic_number=self.max_atomic_number,
            properties={
                "formation_energy_per_atom": torch.tensor(
                    float(row["formation_energy_per_atom"]), dtype=torch.float32
                ),
                "band_gap": torch.tensor(float(row["band_gap"]), dtype=torch.float32),
                "e_above_hull": torch.tensor(float(row["e_above_hull"]), dtype=torch.float32),
                "spacegroup_number": torch.tensor(
                    float(row.get("spacegroup.number", 0.0) or 0.0),
                    dtype=torch.float32,
                ),
            },
        )
        if self._item_cache is not None:
            self._item_cache[idx] = item
        return item


def collate_crystal_batch(items: Iterable[dict[str, object]]) -> CrystalBatch:
    items = list(items)
    pos, x, batch, atomic_numbers = [], [], [], []
    cells, pbc, y, num_atoms, material_ids, volumes = [], [], [], [], [], []

    for graph_idx, item in enumerate(items):
        item_x = item["x"]
        item_pos = item["pos"]
        assert isinstance(item_x, Tensor)
        assert isinstance(item_pos, Tensor)

        n = item_x.shape[0]
        x.append(item_x)
        pos.append(item_pos)
        batch.append(torch.full((n,), graph_idx, dtype=torch.long))
        atomic_numbers.append(item["atomic_numbers"])
        cells.append(item["cell"])
        pbc.append(item["pbc"])
        y.append(item["y"])
        num_atoms.append(item["num_atoms"])
        material_ids.append(str(item["material_id"]))
        volumes.append(item["volume"])

    return CrystalBatch(
        pos=torch.cat(pos, dim=0),
        x=torch.cat(x, dim=0),
        y=torch.cat(y, dim=0).view(-1),
        batch=torch.cat(batch, dim=0),
        cell=torch.stack(cells, dim=0),
        pbc=torch.stack(pbc, dim=0),
        num_atoms=torch.stack(num_atoms, dim=0),
        atomic_numbers=torch.cat(atomic_numbers, dim=0),
        material_ids=material_ids,
        volumes=torch.stack(volumes, dim=0),
    )


def split_dataset(
    dataset: Dataset,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    n_val = max(1, int(round(len(dataset) * val_fraction)))
    n_train = len(dataset) - n_val
    if n_train <= 0:
        raise ValueError("Dataset must contain at least two structures for train/val split.")
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(dataset, [n_train, n_val], generator=generator)
    return train, val
