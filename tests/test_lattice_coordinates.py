import math
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from platonic_transformers.models.platoformer.lattice import (
    cartesian_to_fractional,
    fractional_to_cartesian,
)
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.models.platoformer.utils import scatter_add


def _hexagonal_lattice(dtype=torch.float32):
    a = 2.46
    c = 6.70
    return torch.tensor(
        [
            [a, 0.0, 0.0],
            [-0.5 * a, 0.5 * math.sqrt(3.0) * a, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=dtype,
    )


def _center_by_graph(pos, batch):
    num_graphs = int(batch.max()) + 1
    counts = scatter_add(torch.ones_like(pos[:, :1]), batch, dim_size=num_graphs).clamp_min(1.0)
    mean = scatter_add(pos, batch, dim_size=num_graphs) / counts
    return pos - mean[batch]


def test_fractional_cartesian_roundtrip_with_batched_lattice():
    torch.manual_seed(0)
    batch = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    lattice = _hexagonal_lattice().expand(3, -1, -1).contiguous()
    frac = torch.rand(batch.numel(), 3)

    cart = fractional_to_cartesian(frac, lattice, batch=batch)
    recovered = cartesian_to_fractional(cart, lattice, batch=batch)
    shared_cart = fractional_to_cartesian(frac, lattice[:1], batch=batch)

    assert torch.allclose(frac, recovered, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cart, shared_cart, atol=1e-6, rtol=1e-6)


def test_platonic_transformer_fractional_lattice_matches_cartesian():
    torch.manual_seed(0)
    model = PlatonicTransformer(
        input_dim=11,
        input_dim_vec=0,
        hidden_dim=48,
        output_dim=1,
        output_dim_vec=0,
        nhead=12,
        num_layers=1,
        solid_name="tetrahedron",
        spatial_dim=3,
        scalar_task_level="graph",
        vector_task_level="graph",
        attention=True,
        dropout=0.0,
        rope_sigma=1.5,
        ape_sigma=0.5,
        learned_freqs=True,
        freq_init="spiral",
        rope_on_values=True,
        attention_backend="scatter",
    ).eval()

    x = torch.randn(10, 11)
    batch = torch.tensor([0] * 4 + [1] * 6)
    cart = _center_by_graph(torch.randn(10, 3), batch)
    lattice = _hexagonal_lattice().expand(2, -1, -1).contiguous()
    frac = cartesian_to_fractional(cart, lattice, batch=batch)

    with torch.no_grad():
        cart_out, _ = model(x, cart, batch=batch, avg_num_nodes=torch.tensor(5.0))
        frac_out, _ = model(
            x,
            frac,
            batch=batch,
            lattice=lattice,
            fractional_pos=True,
            avg_num_nodes=torch.tensor(5.0),
        )

    assert torch.allclose(cart_out, frac_out, atol=1e-6, rtol=1e-6)
