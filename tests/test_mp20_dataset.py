import math
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

pymatgen = pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure

from platonic_transformers.datasets.mp20 import MP20CIFDataset, collate_crystal_batch


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
