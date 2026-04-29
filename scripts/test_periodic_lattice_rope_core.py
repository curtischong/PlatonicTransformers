#!/usr/bin/env python
"""Minimal sanity check for periodic lattice RoPE.

The core idea is to apply RoPE to fractional displacements with integer
harmonics. Integer lattice translations then change phases by whole turns, so
the rotation is unchanged.
"""

import argparse
import math

import torch
from pymatgen.core import Lattice


INTEGER_MODES = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
)


def lattice_matrix(device: torch.device) -> torch.Tensor:
    lattice = Lattice.hexagonal(a=2.46, c=6.70)
    return torch.tensor(lattice.matrix, dtype=torch.float32, device=device)


def fractional_displacement(cart_displacement: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Solve ``cart_displacement = frac_displacement @ lattice``."""
    flat = cart_displacement.reshape(-1, 3).double()
    frac = torch.linalg.solve(lattice.double().T, flat.T).T
    return frac.reshape_as(cart_displacement).float()


def wrapped_phase(frac_displacement: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    phase = torch.einsum("...d,md->...m", frac_displacement.float(), modes)
    return phase - torch.floor(phase + 0.5)


def periodic_rope(x: torch.Tensor, frac_displacement: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
    # Integer modes make this phase periodic on the unit fractional cell:
    # frac_displacement + integer_shift gives the same angle modulo 2*pi.
    angle = 2.0 * math.pi * wrapped_phase(frac_displacement, modes)

    # Standard RoPE rotates each adjacent feature pair by the corresponding
    # periodic lattice angle.
    x0, x1 = x.view(*x.shape[:-1], modes.shape[0], 2).unbind(-1)
    y0 = x0 * angle.cos() - x1 * angle.sin()
    y1 = x0 * angle.sin() + x1 * angle.cos()
    return torch.stack((y0, y1), dim=-1).flatten(-2)


def check_integer_translation_invariance(device: torch.device, modes: torch.Tensor) -> None:
    # Use a real non-orthogonal pymatgen lattice so fractional and Cartesian
    # displacements are not interchangeable by accident.
    lattice = lattice_matrix(device)
    frac = torch.rand(5, 3, device=device)
    cart = frac @ lattice

    # Pairwise Cartesian displacements should recover the original fractional
    # displacements after solving against the lattice matrix.
    frac_disp = fractional_displacement(cart[None, :, :] - cart[:, None, :], lattice)
    expected = frac[None, :, :] - frac[:, None, :]
    roundtrip_err = (frac_disp - expected).abs().max().item()

    # Shift one atom by an integer number of unit cells. This changes its
    # Cartesian coordinate but should not change periodic fractional phases.
    shifted_frac = frac.clone()
    shifted_frac[0] += torch.tensor([1.0, -2.0, 1.0], device=device)
    shifted_cart = shifted_frac @ lattice
    shifted_frac_disp = fractional_displacement(
        shifted_cart[None, :, :] - shifted_cart[:, None, :],
        lattice,
    )

    # The RoPE input is arbitrary pair features. Only the fractional
    # displacement phases should decide whether the rotations agree.
    x = torch.randn(*frac_disp.shape[:-1], 2 * modes.shape[0], device=device)
    phase_err = (wrapped_phase(frac_disp, modes) - wrapped_phase(shifted_frac_disp, modes)).abs().max().item()
    rope_err = (periodic_rope(x, frac_disp, modes) - periodic_rope(x, shifted_frac_disp, modes)).abs().max().item()

    print(f"cart->frac roundtrip error: {roundtrip_err:.3e}")
    print(f"integer translation phase error: {phase_err:.3e}")
    print(f"integer translation RoPE error: {rope_err:.3e}")
    assert roundtrip_err < 1e-5
    assert phase_err < 1e-5
    assert rope_err < 2e-5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    modes = torch.tensor(INTEGER_MODES, dtype=torch.float32, device=device)
    print(f"device={device}")

    check_integer_translation_invariance(device, modes)
    print("periodic lattice RoPE core checks passed")


if __name__ == "__main__":
    main()
