"""Patchifier modules for the Platonic Transformer.

A patchifier converts raw point-cloud inputs into the format expected by the
encoder body. The interface is uniform:

    patchifier(x, vec, pos, batch) -> (features, pos, mask, batch)

Exactly one of `mask` and `batch` is non-`None` in the output:
- dense outputs: `features` is `(B, N, hidden_dim)`, `mask` is `(B, N)`,
  `batch` is `None`.
- sparse outputs: `features` is `(N_total, hidden_dim)`, `batch` is `(N_total,)`,
  `mask` is `None`.

Two flavors are provided:
- `StandardPatchifier`: dataloader is assumed to have done FPS+KNN already
  (e.g. with the `patchify` helper that stacks K offset vectors per center).
  This module just lifts, embeds, and adds APE. Honors `dense_mode` —
  returns dense if true, sparse if false.
- `EdgeConvPatchifier`: in-model FPS + KNN + `PlatonicEdgeConv` (per-edge MLP
  + max-pool, equivariant). Always returns dense over the FPS centers.

By extracting these into a uniform interface, `PlatonicTransformer.forward`
can delegate the entire input-adaptation step. Custom patchifiers (CV grid
embeddings, ProteinMD upsamplers, etc.) can plug in via the constructor's
`patchifier=` argument as long as they conform to the same signature.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from platonic_transformers.models.platoformer.ape import PlatonicAPE as APE
from platonic_transformers.models.platoformer.conv import PlatonicEdgeConv
from platonic_transformers.models.platoformer.io import lift, to_dense_and_mask
from platonic_transformers.models.platoformer.linear import PlatonicLinear


@torch.compiler.disable
def _fps_no_compile(pos, batch, ratio):
    """torch_cluster.fps wrapper kept out of dynamo tracing."""
    import torch_geometric.nn as tgnn
    return tgnn.fps(pos, batch, ratio)


@torch.compiler.disable
def _knn_no_compile(x, y, batch_x, batch_y, k):
    """torch_cluster.knn wrapper kept out of dynamo tracing."""
    import torch_geometric.nn as tgnn
    return tgnn.knn(x=x, y=y, batch_x=batch_x, batch_y=batch_y, k=k)


def _build_x_embedder(group, input_dim: int, input_dim_vec: int, hidden_dim: int,
                      spatial_dim: int, solid_name: str) -> PlatonicLinear:
    return PlatonicLinear(
        (input_dim + input_dim_vec * spatial_dim) * group.G,
        hidden_dim,
        solid_name,
        bias=False,
    )


class StandardPatchifier(nn.Module):
    """Lift + embed + APE. Honors `dense_mode` — converts sparse → dense if
    true, otherwise passes the sparse format through.
    """

    def __init__(
        self,
        group,
        input_dim: int,
        input_dim_vec: int,
        hidden_dim: int,
        spatial_dim: int,
        solid_name: str,
        ape_sigma: Optional[float],
        learned_freqs: bool,
        dense_mode: bool,
    ) -> None:
        super().__init__()
        self.group = group
        self.dense_mode = dense_mode
        self.x_embedder = _build_x_embedder(group, input_dim, input_dim_vec,
                                            hidden_dim, spatial_dim, solid_name)
        if ape_sigma is not None:
            self.ape = APE(hidden_dim, solid_name, ape_sigma, spatial_dim, learned_freqs)
        else:
            self.register_buffer('ape', None)
        # Whether the *raw* caller input was already dense — used by node-level
        # readout in PlatonicTransformer to know when to apply mask-indexing.
        self.input_was_dense_format: bool = False

    def forward(
        self,
        x: Optional[Tensor],
        vec: Optional[Tensor],
        pos: Tensor,
        batch: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if self.dense_mode:
            self.input_was_dense_format = (batch is None)
            x, vec, pos, mask = to_dense_and_mask(x, vec, pos, batch)
            out_batch = None
        else:
            self.input_was_dense_format = False
            mask = None
            out_batch = batch
        x_lifted = lift(x, vec, self.group)
        x_emb = self.x_embedder(x_lifted)
        if self.ape is not None:
            x_emb = x_emb + self.ape(pos)
        return x_emb, pos, mask, out_batch


class EdgeConvPatchifier(nn.Module):
    """In-model FPS + KNN + `PlatonicEdgeConv` patchification. Sparse input
    required. Always returns dense `(features, pos, mask, batch=None)` over
    the FPS centers.
    """

    def __init__(
        self,
        group,
        input_dim: int,
        input_dim_vec: int,
        hidden_dim: int,
        spatial_dim: int,
        solid_name: str,
        ape_sigma: Optional[float],
        learned_freqs: bool,
        ratio: float,
        k: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.group = group
        self.ratio = ratio
        self.k = k
        self.x_embedder = _build_x_embedder(group, input_dim, input_dim_vec,
                                            hidden_dim, spatial_dim, solid_name)
        if ape_sigma is not None:
            self.ape = APE(hidden_dim, solid_name, ape_sigma, spatial_dim, learned_freqs)
        else:
            self.register_buffer('ape', None)
        self.edge_conv = PlatonicEdgeConv(
            kv_in_channels=hidden_dim,
            out_channels=hidden_dim,
            embed_dim=hidden_dim,
            solid_name=solid_name,
            spatial_dims=spatial_dim,
            bias=True,
            activation=activation,
        )
        self.input_was_dense_format: bool = False

    def forward(
        self,
        x: Optional[Tensor],
        vec: Optional[Tensor],
        pos: Tensor,
        batch: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if batch is None:
            raise ValueError("EdgeConvPatchifier requires sparse input (batch != None).")
        x_lifted = lift(x, vec, self.group)
        kv_x = self.x_embedder(x_lifted)
        if self.ape is not None:
            kv_x = kv_x + self.ape(pos)
        center_ids = _fps_no_compile(pos, batch, self.ratio)
        q_pos_sparse = pos[center_ids]
        q_batch_sparse = batch[center_ids]
        knn_edges = _knn_no_compile(pos, q_pos_sparse, batch, q_batch_sparse, self.k)
        n_centers = q_pos_sparse.shape[0]
        knn_indices = knn_edges[1].view(n_centers, self.k)
        x_centers = self.edge_conv(q_pos_sparse, kv_x, pos, knn_indices)
        x_dense, _, pos_dense, mask = to_dense_and_mask(
            x_centers, None, q_pos_sparse, q_batch_sparse
        )
        return x_dense, pos_dense, mask, None
