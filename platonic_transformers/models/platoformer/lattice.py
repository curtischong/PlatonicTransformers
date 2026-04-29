from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def canonicalize_lattice(
    lattice: Optional[Tensor],
    num_graphs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    """Return lattice cells as ``[B, D, D]`` row-vector cell matrices.

    Positions in this repository use Cartesian coordinates. For periodic data,
    ``lattice`` is expected to follow the ASE/PyG convention where Cartesian
    coordinates are obtained from fractional coordinates with ``frac @ cell``.
    """
    if lattice is None:
        return None

    lattice = torch.as_tensor(lattice, device=device, dtype=dtype)
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    if lattice.ndim != 3 or lattice.shape[-1] != lattice.shape[-2]:
        raise ValueError(
            "lattice must have shape [D, D] or [B, D, D], "
            f"got {tuple(lattice.shape)}"
        )
    if lattice.shape[0] == 1 and num_graphs > 1:
        lattice = lattice.expand(num_graphs, -1, -1)
    if lattice.shape[0] != num_graphs:
        raise ValueError(
            f"lattice batch dimension ({lattice.shape[0]}) must be 1 or {num_graphs}"
        )
    return lattice


def canonicalize_pbc(
    pbc: Optional[Tensor],
    num_graphs: int,
    spatial_dim: int,
    device: torch.device,
) -> Tensor:
    """Return periodic-boundary flags as ``[B, D]`` booleans.

    When a lattice is supplied and no explicit ``pbc`` is given, all lattice
    directions are treated as periodic.
    """
    if pbc is None:
        return torch.ones(num_graphs, spatial_dim, dtype=torch.bool, device=device)

    pbc = torch.as_tensor(pbc, device=device, dtype=torch.bool)
    if pbc.ndim == 0:
        pbc = pbc.expand(spatial_dim)
    if pbc.ndim == 1:
        if pbc.shape[0] != spatial_dim:
            raise ValueError(
                f"pbc must have {spatial_dim} entries, got {pbc.shape[0]}"
            )
        pbc = pbc.unsqueeze(0)
    if pbc.ndim != 2 or pbc.shape[-1] != spatial_dim:
        raise ValueError(
            f"pbc must have shape [{spatial_dim}] or [B, {spatial_dim}], "
            f"got {tuple(pbc.shape)}"
        )
    if pbc.shape[0] == 1 and num_graphs > 1:
        pbc = pbc.expand(num_graphs, -1)
    if pbc.shape[0] != num_graphs:
        raise ValueError(f"pbc batch dimension ({pbc.shape[0]}) must be 1 or {num_graphs}")
    return pbc
