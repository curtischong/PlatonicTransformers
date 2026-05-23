"""Tests for PlatonicAPE.

The frequency initialization is intentionally done under a CUDA default-device
context so the random frequencies are controlled by the CUDA RNG seed.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from platonic_transformers.models.platoformer.ape import PlatonicAPE
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PlatonicAPE CUDA seed tests require CUDA.",
)


def _make_cuda_ape(
    solid_name: str,
    seed: int,
    *,
    embed_dim_per_group: int = 4,
    freq_sigma: float = 1.25,
    learned_freqs: bool = False,
    dtype: torch.dtype = torch.float32,
) -> PlatonicAPE:
    torch.cuda.manual_seed_all(seed)
    group = PLATONIC_GROUPS[solid_name]
    embed_dim = group.G * embed_dim_per_group

    with torch.device("cuda"):
        ape = PlatonicAPE(
            embed_dim=embed_dim,
            solid_name=solid_name,
            freq_sigma=freq_sigma,
            spatial_dims=group.dim,
            learned_freqs=learned_freqs,
        )

    return ape.to(device="cuda", dtype=dtype)


def _cuda_randn(seed: int, *shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    torch.cuda.manual_seed_all(seed)
    return torch.randn(*shape, device="cuda", dtype=dtype)


@pytest.mark.parametrize("learned_freqs", [False, True])
def test_cuda_seed_controls_frequency_initialization(learned_freqs):
    torch.manual_seed(1)
    ape_a = _make_cuda_ape("tetrahedron", 1234, learned_freqs=learned_freqs)

    torch.manual_seed(9999)
    ape_b = _make_cuda_ape("tetrahedron", 1234, learned_freqs=learned_freqs)
    ape_c = _make_cuda_ape("tetrahedron", 1235, learned_freqs=learned_freqs)

    assert ape_a.freqs.is_cuda
    torch.testing.assert_close(ape_a.freqs, ape_b.freqs, rtol=0, atol=0)
    assert not torch.equal(ape_a.freqs, ape_c.freqs)


def test_cuda_default_device_constructor_keeps_buffers_on_cuda():
    torch.cuda.manual_seed_all(5678)
    group = PLATONIC_GROUPS["tetrahedron"]

    with torch.device("cuda"):
        ape = PlatonicAPE(
            embed_dim=group.G * 4,
            solid_name="tetrahedron",
            freq_sigma=1.25,
        )
        pos = torch.randn(3, group.dim)

    assert ape.freqs.is_cuda
    assert ape.group_elements.is_cuda
    assert ape(pos).is_cuda


@pytest.mark.parametrize("solid_name", ["tetrahedron", "octahedron", "icosahedron"])
def test_forward_matches_grouped_sinusoidal_reference(solid_name):
    ape = _make_cuda_ape(solid_name, 2026)
    pos = _cuda_randn(17, 2, 5, ape.spatial_dims)

    out = ape(pos)

    freqs_rotated = torch.einsum("gij,jf->gif", ape.group_elements, ape.freqs)
    angles = torch.einsum("...d,gdf->...gf", pos, freqs_rotated)
    expected = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    expected = expected.reshape(*pos.shape[:-1], ape.embed_dim)

    assert out.shape == (*pos.shape[:-1], ape.embed_dim)
    assert out.device.type == "cuda"
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("solid_name", "atol"),
    [
        ("tetrahedron", 1e-6),
        ("octahedron", 1e-6),
        ("icosahedron", 2e-5),
    ],
)
def test_rotating_positions_permutes_group_axis_by_inverse_left_action(solid_name, atol):
    group = PLATONIC_GROUPS[solid_name]
    ape = _make_cuda_ape(solid_name, 314159)
    pos = _cuda_randn(271828, 7, group.dim)

    base = ape(pos).reshape(pos.shape[0], group.G, ape.embed_dim_g)
    rotated_pos = torch.einsum("hde,ne->hnd", ape.group_elements, pos)
    rotated = ape(rotated_pos).reshape(group.G, pos.shape[0], group.G, ape.embed_dim_g)

    for h in range(group.G):
        expected_indices = group.cayley_table[group.inverse_indices[h], :].to("cuda")
        expected = base[:, expected_indices, :]
        torch.testing.assert_close(
            rotated[h],
            expected,
            rtol=1e-5,
            atol=atol,
            msg=(
                f"{solid_name} failed for group element {h}; "
                "expected APE(h @ pos)[g] == APE(pos)[h^-1 * g]"
            ),
        )


def test_to_dtype_keeps_forward_dtype():
    ape = _make_cuda_ape("tetrahedron", 42, dtype=torch.float64)
    pos = _cuda_randn(43, 3, ape.spatial_dims, dtype=torch.float64)

    out = ape(pos)

    assert ape.freqs.dtype == torch.float64
    assert ape.group_elements.dtype == torch.float64
    assert out.dtype == torch.float64


@pytest.mark.parametrize(
    ("embed_dim", "solid_name", "match"),
    [
        (13, "tetrahedron", "must be divisible by group size"),
        (36, "tetrahedron", "must be an even number"),
        (12, "not_a_group", "Unknown solid"),
    ],
)
def test_invalid_configuration_errors(embed_dim, solid_name, match):
    with pytest.raises(ValueError, match=match):
        PlatonicAPE(embed_dim, solid_name, freq_sigma=1.0)
