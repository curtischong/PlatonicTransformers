"""eSEN (Smooth Energy Network) wrapper around fairchem's eSCN-MD backbone.

The backbone is ``fairchem.core.models.uma.escn_md.eSCNMDBackbone`` — the
lmax/mmax spherical-harmonic equivariant message-passing network shared by the
UMA family. We attach ``MLP_EFS_Head`` with ``direct_forces=False`` so that
forces are obtained as autograd derivatives of the predicted energy (the
"smooth energy" recipe that makes this *eSEN* rather than *UMA-direct*).

Interface
---------
``forward(data) -> {"energy": (B,), "forces": (N, 3)}``

``data`` is a dict (or PyG ``Data``) with keys:
``pos``, ``batch``, ``atomic_numbers``, optionally ``charge``, ``spin``,
``natoms``, ``cell``, ``pbc``, ``edge_index``, ``cell_offsets``.

Requires ``trainer.inference_mode=false`` at val/test because forces use
``torch.autograd.grad`` and therefore need grad-enabled inference.

Optional dependency
-------------------
``fairchem-core>=2.19`` is required to import the backbone. Install with::

    pip install fairchem-core==2.19
"""
from typing import List, Optional

import torch
import torch.nn as nn

try:
    from fairchem.core.models.uma.escn_md import (
        Linear_Energy_Head,
        Linear_Force_Head,
        MLP_EFS_Head,
        eSCNMDBackbone,
    )
    _FAIRCHEM_AVAILABLE = True
except ImportError:
    _FAIRCHEM_AVAILABLE = False


class EquivariantNet(nn.Module):
    """eSEN backbone + EFS head with autograd-derivative forces.

    Designed to drop into the Lightning ``ESENModel`` in
    ``mains/main_omol.py`` for OMol25 training. See ``configs/omol_esen.yaml``
    for the eSEN-small recipe defaults.
    """

    def __init__(
        self,
        # ---- eSCNMDBackbone passthrough ----
        max_num_elements: int = 100,
        sphere_channels: int = 128,
        lmax: int = 2,
        mmax: int = 2,
        grid_resolution: Optional[int] = None,
        num_sphere_samples: int = 128,
        otf_graph: bool = True,
        max_neighbors: int = 30,
        cutoff: float = 6.0,
        edge_channels: int = 128,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        direct_forces: bool = False,
        regress_forces: bool = True,
        regress_stress: bool = False,
        num_layers: int = 2,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        ff_type: str = "grid",
        activation_checkpointing: bool = False,
        chg_spin_emb_type: str = "pos_emb",
        cs_emb_grad: bool = False,
        dataset_emb_grad: bool = False,
        dataset_list: Optional[List[str]] = None,
        use_dataset_embedding: bool = False,
        use_cuda_graph_wigner: bool = False,
        radius_pbc_version: int = 2,
        always_use_pbc: bool = False,
        use_pbc: bool = False,
        use_pbc_single: bool = False,
        # ---- our extras ----
        avg_num_nodes: float = 1.0,
    ):
        super().__init__()
        if not _FAIRCHEM_AVAILABLE:
            raise ImportError(
                "EquivariantNet requires fairchem-core. Install with "
                "`pip install fairchem-core>=2.19` (or use the [omol] extras "
                "of this package once published)."
            )
        self.backbone = eSCNMDBackbone(
            max_num_elements=max_num_elements,
            sphere_channels=sphere_channels,
            lmax=lmax, mmax=mmax,
            grid_resolution=grid_resolution,
            num_sphere_samples=num_sphere_samples,
            otf_graph=otf_graph,
            max_neighbors=max_neighbors,
            use_pbc=use_pbc,
            use_pbc_single=use_pbc_single,
            cutoff=cutoff,
            edge_channels=edge_channels,
            distance_function=distance_function,
            num_distance_basis=num_distance_basis,
            direct_forces=direct_forces,
            regress_forces=regress_forces,
            regress_stress=regress_stress,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            norm_type=norm_type,
            act_type=act_type,
            ff_type=ff_type,
            activation_checkpointing=activation_checkpointing,
            chg_spin_emb_type=chg_spin_emb_type,
            cs_emb_grad=cs_emb_grad,
            dataset_emb_grad=dataset_emb_grad,
            dataset_list=dataset_list,
            use_dataset_embedding=use_dataset_embedding,
            use_cuda_graph_wigner=use_cuda_graph_wigner,
            radius_pbc_version=radius_pbc_version,
            always_use_pbc=always_use_pbc,
        )
        # Conservative (MLP_EFS_Head): forces via autograd.grad through energy.
        # Direct: separate Linear_Force_Head reads an L=1 component from the
        # node embedding (avoids the double-backward when training).
        self.direct_forces = direct_forces
        if direct_forces:
            self.energy_head = Linear_Energy_Head(self.backbone, reduce="sum")
            self.force_head = Linear_Force_Head(self.backbone)
        else:
            self.head = MLP_EFS_Head(
                self.backbone, reduce="sum", prefix=None, wrap_property=False,
            )
        self.avg_num_nodes = avg_num_nodes

    def forward(self, data):
        if not self.direct_forces:
            pos = data["pos"] if isinstance(data, dict) else data.pos
            if not pos.requires_grad:
                pos.requires_grad_(True)

        emb = self.backbone(data)
        if self.direct_forces:
            e_out = self.energy_head(data, emb)
            f_out = self.force_head(data, emb)
            return {"energy": e_out["energy"].view(-1), "forces": f_out["forces"]}
        out = self.head(data, emb)
        return {"energy": out["energy"].view(-1), "forces": out["forces"]}
