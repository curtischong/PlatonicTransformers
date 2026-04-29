#!/usr/bin/env python
"""Standalone math checks for strict periodic lattice RoPE.

This script does not import PlatonicTransformer and does not train a model. It
only checks the algebra needed by a lattice-aware periodic RoPE layer:

1. Cartesian coordinates are produced by ``cart = frac @ lattice`` for skewed
   row-vector cells.
2. Cartesian pair displacements can be solved back to fractional displacements.
3. Integer cell translations leave modulo-one RoPE rotations unchanged.
4. The same fractional coordinates can have different metric distances under
   different lattices, so lattice parameters must be passed to the network
   separately from periodic phases.
"""

from __future__ import annotations

import argparse
import math

import torch


def make_triclinic_lattice(batch_size: int, device: torch.device) -> torch.Tensor:
    """Create non-orthogonal row-vector cell matrices with positive volume."""
    lengths = 1.6 + 2.4 * torch.rand(batch_size, 3, device=device)
    shear = -0.45 + 0.9 * torch.rand(batch_size, 3, device=device)

    lattice = torch.zeros(batch_size, 3, 3, device=device)
    lattice[:, 0, 0] = lengths[:, 0]
    lattice[:, 1, 0] = shear[:, 0] * lengths[:, 1]
    lattice[:, 1, 1] = lengths[:, 1] * torch.sqrt(1.0 - shear[:, 0].square()).clamp_min(0.35)
    lattice[:, 2, 0] = shear[:, 1] * lengths[:, 2]
    lattice[:, 2, 1] = shear[:, 2] * lengths[:, 2]
    lattice[:, 2, 2] = lengths[:, 2] * 0.75
    return lattice


def frac_to_cart(frac: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bnd,bdc->bnc", frac, lattice)


def fractional_displacement(displacement: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Solve ``disp = frac @ lattice`` for fractional displacements."""
    return torch.linalg.solve(
        lattice.double().transpose(-1, -2),
        displacement.double().unsqueeze(-1),
    ).squeeze(-1).float()


def lattice_features(lattice: torch.Tensor) -> torch.Tensor:
    """Rotation-invariant lattice parameters a scalar network can consume."""
    lengths = torch.linalg.norm(lattice, dim=-1).clamp_min(1e-8)
    cos_alpha = (lattice[:, 1] * lattice[:, 2]).sum(-1) / (lengths[:, 1] * lengths[:, 2])
    cos_beta = (lattice[:, 0] * lattice[:, 2]).sum(-1) / (lengths[:, 0] * lengths[:, 2])
    cos_gamma = (lattice[:, 0] * lattice[:, 1]).sum(-1) / (lengths[:, 0] * lengths[:, 1])
    volume = torch.linalg.det(lattice).abs().clamp_min(1e-8)
    return torch.stack(
        [lengths[:, 0], lengths[:, 1], lengths[:, 2], cos_alpha, cos_beta, cos_gamma, torch.log(volume)],
        dim=-1,
    )


def make_modes(num_heads: int, num_pairs: int, device: torch.device) -> torch.Tensor:
    candidates = []
    radius = 1
    while len(candidates) < num_heads * num_pairs:
        shell = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                for k in range(-radius, radius + 1):
                    if (i, j, k) != (0, 0, 0) and max(abs(i), abs(j), abs(k)) == radius:
                        shell.append((i, j, k))
        shell.sort(key=lambda v: (sum(x * x for x in v), sum(abs(x) for x in v), v))
        candidates.extend(shell)
        radius += 1
    return torch.tensor(candidates[: num_heads * num_pairs], dtype=torch.float32, device=device).view(
        num_heads,
        num_pairs,
        3,
    )


def wrapped_phase(frac: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    phase = torch.einsum("...d,hfd->...hf", frac.float(), modes)
    return phase - torch.floor(phase + 0.5)


def apply_periodic_rope(x: torch.Tensor, frac: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    """Apply RoPE with modulo-one integer lattice harmonics."""
    *leading, num_heads, head_dim = x.shape
    num_pairs = head_dim // 2
    angle = wrapped_phase(frac, modes) * (2.0 * math.pi)
    cos = angle.cos()
    sin = angle.sin()

    x0, x1 = x.view(*leading, num_heads, num_pairs, 2).unbind(-1)
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack([y0, y1], dim=-1).view(*leading, num_heads, head_dim)


def brute_force_periodic_distances(frac: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """True nearest-image distances by enumerating neighboring cells.

    Component-wise rounding in fractional coordinates is not generally the
    shortest image for skewed cells, so this explicitly checks the 27 closest
    cell translations.
    """
    shifts = torch.tensor(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=frac.dtype,
        device=frac.device,
    )
    delta = frac[:, :, None, :] - frac[:, None, :, :]
    images = delta[:, :, :, None, :] + shifts
    cart_images = torch.einsum("bijnc,bdc->bijnd", images, lattice)
    return torch.linalg.norm(cart_images, dim=-1).amin(dim=-1)


def check_triclinic_roundtrip(batch_size: int, num_atoms: int, device: torch.device) -> None:
    lattice = make_triclinic_lattice(batch_size, device)
    frac = torch.rand(batch_size, num_atoms, 3, device=device)
    cart = frac_to_cart(frac, lattice)

    displacement = cart[:, None, :, :] - cart[:, :, None, :]
    recovered = fractional_displacement(displacement, lattice[:, None, None, :, :])
    expected = frac[:, None, :, :] - frac[:, :, None, :]
    err = (recovered - expected).abs().max().item()

    off_diag_idx = torch.tril_indices(3, 3, offset=-1, device=device)
    off_diagonal = lattice[:, off_diag_idx[0], off_diag_idx[1]].abs().max().item()
    print(f"triclinic roundtrip max error: {err:.3e}")
    print(f"max off-diagonal lattice term: {off_diagonal:.3f}")
    assert off_diagonal > 0.05
    assert err < 1e-5


def check_integer_shift_rope_invariance(batch_size: int, num_atoms: int, device: torch.device) -> None:
    lattice = make_triclinic_lattice(batch_size, device)
    frac = torch.rand(batch_size, num_atoms, 3, device=device)
    cart = frac_to_cart(frac, lattice)

    modes = make_modes(num_heads=3, num_pairs=4, device=device)
    pair_features = torch.randn(batch_size, num_atoms, num_atoms, 3, 8, device=device)

    displacement = cart[:, None, :, :] - cart[:, :, None, :]
    frac_disp = fractional_displacement(displacement, lattice[:, None, None, :, :])
    rotated = apply_periodic_rope(pair_features, frac_disp, modes)

    shifted = cart.clone()
    integer_shift = torch.tensor([1.0, -2.0, 1.0], device=device)
    shifted[:, 0] = shifted[:, 0] + torch.einsum("d,bdc->bc", integer_shift, lattice)
    shifted_displacement = shifted[:, None, :, :] - shifted[:, :, None, :]
    shifted_frac_disp = fractional_displacement(shifted_displacement, lattice[:, None, None, :, :])
    shifted_rotated = apply_periodic_rope(pair_features, shifted_frac_disp, modes)

    phase_err = (wrapped_phase(frac_disp, modes) - wrapped_phase(shifted_frac_disp, modes)).abs().max().item()
    rope_err = (rotated - shifted_rotated).abs().max().item()
    print(f"integer shift wrapped phase max error: {phase_err:.3e}")
    print(f"integer shift RoPE max error: {rope_err:.3e}")
    assert phase_err < 1e-5
    assert rope_err < 2e-5


def check_lattice_parameters_are_needed(batch_size: int, num_atoms: int, device: torch.device) -> None:
    frac = torch.rand(batch_size, num_atoms, 3, device=device)
    lattice_a = make_triclinic_lattice(batch_size, device)
    lattice_b = make_triclinic_lattice(batch_size, device)
    cart_a = frac_to_cart(frac, lattice_a)
    cart_b = frac_to_cart(frac, lattice_b)

    modes = make_modes(num_heads=2, num_pairs=3, device=device)
    frac_disp_a = fractional_displacement(
        cart_a[:, None, :, :] - cart_a[:, :, None, :],
        lattice_a[:, None, None, :, :],
    )
    frac_disp_b = fractional_displacement(
        cart_b[:, None, :, :] - cart_b[:, :, None, :],
        lattice_b[:, None, None, :, :],
    )
    phase_delta = (wrapped_phase(frac_disp_a, modes) - wrapped_phase(frac_disp_b, modes)).abs().max().item()

    dist_a = brute_force_periodic_distances(frac, lattice_a)
    dist_b = brute_force_periodic_distances(frac, lattice_b)
    distance_delta = (dist_a - dist_b).abs().mean().item()
    feature_delta = (lattice_features(lattice_a) - lattice_features(lattice_b)).abs().mean().item()
    target_a = torch.exp(-dist_a).mean(dim=(1, 2))
    target_b = torch.exp(-dist_b).mean(dim=(1, 2))
    target_delta = (target_a - target_b).abs().mean().item()

    print(f"same fractional coords wrapped phase delta across lattices: {phase_delta:.3e}")
    print(f"same fractional coords metric distance mean delta: {distance_delta:.3f}")
    print(f"lattice feature mean delta: {feature_delta:.3f}")
    print(f"distance-based target mean delta: {target_delta:.3f}")
    assert phase_delta < 1e-5
    assert distance_delta > 0.10
    assert feature_delta > 0.10
    assert target_delta > 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-atoms", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device}")

    check_triclinic_roundtrip(args.batch_size, args.num_atoms, device)
    check_integer_shift_rope_invariance(args.batch_size, args.num_atoms, device)
    check_lattice_parameters_are_needed(args.batch_size, args.num_atoms, device)
    print("all periodic lattice RoPE math checks passed")


if __name__ == "__main__":
    main()
