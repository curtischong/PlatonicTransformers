from typing import Optional

import torch
from torch import Tensor


def _prepare_lattice_for_pos(pos: Tensor, lattice: Tensor, batch: Optional[Tensor] = None) -> Tensor:
    """Broadcast a row-vector lattice tensor to the leading dimensions of ``pos``."""
    if lattice is None:
        raise ValueError("lattice must be provided when fractional_pos=True.")

    lattice = lattice.to(device=pos.device, dtype=pos.dtype)
    spatial_dim = pos.shape[-1]
    if lattice.shape[-2:] != (spatial_dim, spatial_dim):
        raise ValueError(
            f"lattice trailing shape must be ({spatial_dim}, {spatial_dim}), "
            f"got {tuple(lattice.shape)}."
        )

    if lattice.ndim == 2:
        return lattice

    if lattice.ndim != 3:
        raise ValueError(
            "lattice must have shape (D, D) or (B, D, D); "
            f"got {tuple(lattice.shape)}."
        )

    if lattice.shape[0] == 1:
        return lattice[0]

    if batch is not None and pos.ndim == 2:
        if batch.shape[0] != pos.shape[0]:
            raise ValueError(
                f"batch length ({batch.shape[0]}) must match number of positions ({pos.shape[0]})."
            )
        lattice = lattice[batch.to(device=lattice.device)]
    elif pos.shape[0] != lattice.shape[0]:
        raise ValueError(
            "batched lattice shape must match the first position dimension, "
            "or a graph-mode batch vector must be provided."
        )

    while lattice.ndim < pos.ndim + 1:
        lattice = lattice.unsqueeze(1)
    return lattice


def fractional_to_cartesian(pos: Tensor, lattice: Tensor, batch: Optional[Tensor] = None) -> Tensor:
    """Convert fractional coordinates to Cartesian coordinates.

    The lattice convention matches pymatgen and the existing PBC baseline code:
    row-vector fractional coordinates are multiplied by row-vector lattice
    matrices, i.e. ``cart = frac @ lattice``.
    """
    lattice = _prepare_lattice_for_pos(pos, lattice, batch=batch)
    if lattice.ndim == 2:
        return pos @ lattice
    return torch.matmul(pos.unsqueeze(-2), lattice).squeeze(-2)


def cartesian_to_fractional(pos: Tensor, lattice: Tensor, batch: Optional[Tensor] = None) -> Tensor:
    """Convert Cartesian coordinates to fractional coordinates."""
    lattice = _prepare_lattice_for_pos(pos, lattice, batch=batch)
    if lattice.ndim == 2:
        flat = pos.reshape(-1, pos.shape[-1])
        frac = torch.linalg.solve(lattice.T, flat.T).T
        return frac.reshape_as(pos)
    return torch.linalg.solve(lattice.transpose(-1, -2), pos.unsqueeze(-1)).squeeze(-1)
