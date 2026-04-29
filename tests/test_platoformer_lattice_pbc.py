import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from platonic_transformers.models.platoformer.conv import PlatonicConv
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.models.platoformer.lattice import minimum_image_displacement
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_minimum_image_displacement_wraps_orthorhombic_and_partial_pbc():
    lattice = torch.eye(3)
    displacement = torch.tensor([[0.90, -0.60, 0.25]])

    wrapped = minimum_image_displacement(displacement, lattice)
    expected = torch.tensor([[-0.10, 0.40, 0.25]])
    assert torch.allclose(wrapped, expected, atol=1e-6)

    partial = minimum_image_displacement(
        displacement,
        lattice,
        pbc=torch.tensor([True, False, True]),
    )
    expected_partial = torch.tensor([[-0.10, -0.60, 0.25]])
    assert torch.allclose(partial, expected_partial, atol=1e-6)


def test_minimum_image_displacement_wraps_triclinic_cells():
    lattice = torch.tensor(
        [
            [2.0, 0.2, 0.0],
            [0.0, 1.5, 0.1],
            [0.0, 0.0, 1.0],
        ]
    )
    frac = torch.tensor([[0.60, -0.55, 0.20]])
    displacement = frac @ lattice
    wrapped = minimum_image_displacement(displacement, lattice)

    expected_frac = torch.tensor([[-0.40, 0.45, 0.20]])
    assert torch.allclose(wrapped, expected_frac @ lattice, atol=1e-6)


def _build_tiny_model(dense_mode=False, output_dim_vec=0):
    return PlatonicTransformer(
        input_dim=4,
        input_dim_vec=0,
        hidden_dim=24,
        output_dim=1,
        output_dim_vec=output_dim_vec,
        nhead=12,
        num_layers=1,
        solid_name="tetrahedron",
        dense_mode=dense_mode,
        scalar_task_level="node",
        vector_task_level="node",
        attention=True,
        rope_sigma=1.5,
        ape_sigma=None,
        learned_freqs=False,
        freq_init="spiral",
        rope_on_values=True,
        dropout=0.0,
    ).eval()


def test_graph_platoformer_is_invariant_to_integer_cell_shifts():
    model = _build_tiny_model()
    x = torch.randn(4, 4)
    pos = torch.tensor(
        [
            [0.05, 0.10, 0.10],
            [0.95, 0.10, 0.10],
            [0.40, 0.70, 0.20],
            [0.35, 0.20, 0.80],
        ],
        dtype=torch.float32,
    )
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    lattice = torch.eye(3).unsqueeze(0)
    pbc = torch.ones(1, 3, dtype=torch.bool)

    shifted = pos.clone()
    shifted[1] += lattice[0, 0]

    with torch.no_grad():
        out, _ = model(x, pos, batch=batch, lattice=lattice, pbc=pbc)
        out_shifted, _ = model(x, shifted, batch=batch, lattice=lattice, pbc=pbc)

    assert torch.allclose(out, out_shifted, atol=1e-5, rtol=1e-5)


def test_dense_platoformer_is_invariant_to_integer_cell_shifts():
    model = _build_tiny_model(dense_mode=True)
    x = torch.randn(1, 4, 4)
    pos = torch.tensor(
        [
            [
                [0.05, 0.10, 0.10],
                [0.95, 0.10, 0.10],
                [0.40, 0.70, 0.20],
                [0.35, 0.20, 0.80],
            ]
        ],
        dtype=torch.float32,
    )
    lattice = torch.eye(3).unsqueeze(0)
    pbc = torch.ones(1, 3, dtype=torch.bool)

    shifted = pos.clone()
    shifted[:, 1] += lattice[0, 0]

    with torch.no_grad():
        out, _ = model(x, pos, lattice=lattice, pbc=pbc)
        out_shifted, _ = model(x, shifted, lattice=lattice, pbc=pbc)

    assert torch.allclose(out, out_shifted, atol=1e-5, rtol=1e-5)


def test_periodic_lattice_path_preserves_platonic_equivariance():
    group = PLATONIC_GROUPS["tetrahedron"]
    model = _build_tiny_model(output_dim_vec=1)
    x = torch.randn(5, 4)
    pos = torch.rand(5, 3) * 0.8 + 0.1
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    lattice = torch.eye(3).unsqueeze(0) * 2.0
    pbc = torch.ones(1, 3, dtype=torch.bool)

    R = group.elements[4].float()
    pos_rot = torch.einsum("ij,nj->ni", R, pos)
    lattice_rot = torch.einsum("ij,bkj->bki", R, lattice)

    with torch.no_grad():
        scalars, vectors = model(x, pos, batch=batch, lattice=lattice, pbc=pbc)
        scalars_rot, vectors_rot = model(
            x,
            pos_rot,
            batch=batch,
            lattice=lattice_rot,
            pbc=pbc,
        )

    expected_vectors = torch.einsum("ij,ncj->nci", R, vectors)
    assert torch.allclose(scalars, scalars_rot, atol=1e-5, rtol=1e-5)
    assert torch.allclose(vectors_rot, expected_vectors, atol=1e-5, rtol=1e-5)


def test_linear_attention_rejects_lattice_because_pbc_is_pairwise():
    conv = PlatonicConv(
        in_channels=8,
        out_channels=8,
        embed_dim=8,
        num_heads=1,
        solid_name="trivial_3",
        spatial_dims=3,
        freq_sigma=1.0,
        learned_freqs=False,
        attention=False,
    )
    x = torch.randn(3, 8)
    pos = torch.rand(3, 3)
    batch = torch.zeros(3, dtype=torch.long)

    with pytest.raises(ValueError, match="lattice/PBC support requires attention=True"):
        conv(x, pos, batch=batch, lattice=torch.eye(3))
