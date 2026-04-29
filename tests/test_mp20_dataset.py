import math
import os
import sys
import csv

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

pymatgen = pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure

from platonic_transformers.datasets.mp20 import (
    MP20CIFDataset,
    MP20CSVRegressionDataset,
    collate_crystal_batch,
)
from mains.main_mp20_sklearn_baseline import item_to_features


def test_mp20_cif_dataset_reads_lattice_and_periodic_batch(tmp_path):
    structure = Structure(
        Lattice.cubic(2.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    structure.to(filename=str(tmp_path / "MP_mp-test.cif"))

    dataset = MP20CIFDataset(tmp_path, target="log_volume_per_atom")
    item = dataset[0]

    assert item["x"].shape == (2, 118)
    assert item["pos"].shape == (2, 3)
    assert item["cell"].shape == (3, 3)
    assert item["pbc"].tolist() == [True, True, True]
    assert torch.isclose(item["y"], torch.tensor([math.log(4.0)])).all()

    batch = collate_crystal_batch([item, item])
    assert batch.x.shape == (4, 118)
    assert batch.pos.shape == (4, 3)
    assert batch.cell.shape == (2, 3, 3)
    assert batch.pbc.shape == (2, 3)
    assert batch.batch.tolist() == [0, 0, 1, 1]
    assert batch.material_ids == ["mp-test", "mp-test"]


def test_mp20_csv_dataset_reads_real_targets_and_cif_strings(tmp_path):
    structure = Structure(
        Lattice.cubic(3.0),
        ["Li", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    csv_path = tmp_path / "train.csv"
    cif = structure.to(fmt="cif")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "",
            "material_id",
            "formation_energy_per_atom",
            "band_gap",
            "pretty_formula",
            "e_above_hull",
            "elements",
            "cif",
            "spacegroup.number",
        ])
        writer.writerow([0, "mp-test", -1.25, 2.5, "LiO", 0.01, "['Li', 'O']", cif, 1])

    dataset = MP20CSVRegressionDataset(csv_path)
    assert len(dataset) == 1
    assert dataset.targets == [-1.25]

    item = dataset[0]
    assert item["material_id"] == "mp-test"
    assert item["x"].shape == (2, 118)
    assert torch.allclose(item["y"], torch.tensor([-1.25]))
    assert torch.allclose(item["band_gap"], torch.tensor(2.5))
    assert torch.allclose(item["spacegroup_number"], torch.tensor(1.0))

    features = item_to_features(item)
    assert features.shape == (249,)
    assert math.isclose(float(features[:118].sum()), 1.0, rel_tol=0.0, abs_tol=1e-6)
