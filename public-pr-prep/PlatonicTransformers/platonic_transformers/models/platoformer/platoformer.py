import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .block import PlatonicBlock
from .groups import PLATONIC_GROUPS
from .linear import PlatonicLinear
from .io import to_dense_and_mask, pool, lift, to_scalars_vectors
from .ape import PlatonicAPE as APE

def _radius_graph_python(pos, r, batch, loop=True, max_num_neighbors=None):
    """Pure-PyTorch fallback for torch_cluster.radius_graph.

    Used when the torch_cluster prebuilt wheel fails to load (e.g. requires a
    newer glibc than the cluster provides). Computes the full N×N distance
    matrix in fp32 and masks by graph + radius. Memory O(N²) — fine for
    typical molecular batches (a few thousand atoms) but inefficient at very
    large N. The function is only invoked inside @torch.compiler.disable, so
    the Python overhead is amortised once per forward.

    Returns edge_index [2, E] with the same convention as
    `torch_cluster.radius_graph(flow='source_to_target')`: row 0 = source
    (neighbour), row 1 = target (center). For a symmetric radius graph both
    directions appear, so this convention is interchangeable for our use.
    """
    N = pos.shape[0]
    d = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)  # [N, N]
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
    within = (d <= r) & same_graph
    if not loop:
        eye = torch.eye(N, dtype=torch.bool, device=pos.device)
        within = within & ~eye
    src, dst = within.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)


# Try the optimised C++ kernel from torch_cluster; fall back to the pure-PyTorch
# implementation if the wheel can't load (most commonly a glibc mismatch on
# RHEL-family compute clusters). The fallback runs in the @torch.compiler.disable
# region so performance impact is bounded.
try:
    from torch_cluster import radius_graph as _radius_graph
except (ImportError, OSError):
    _radius_graph = _radius_graph_python


def _polynomial_cutoff(x: Tensor, r_max: float, p: int = 6) -> Tensor:
    """Klicpera et al. (DimeNet, ICLR 2020) polynomial envelope.

    Smooth window that is 1 at x=0 and 0 at x>=r_max, with continuous derivatives up
    to order p+1 at the boundary. Mirrors `ponita.utils.windowing.PolynomialCutoff`.
    """
    x_over_r = x / r_max
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * x_over_r.pow(p)
        + p * (p + 2.0) * x_over_r.pow(p + 1)
        - (p * (p + 1.0) / 2.0) * x_over_r.pow(p + 2)
    )
    return envelope * (x < r_max).to(x.dtype)


@torch.compiler.disable
def _build_radius_edges(pos: Tensor, batch: Tensor, r_max: float, p: int,
                        max_num_neighbors: int):
    """Build the radius edge set + polynomial cutoff window in a Dynamo-free region.

    torch_cluster.radius is a C++ custom op without a fake-tensor implementation,
    so Dynamo cannot trace it. We isolate the call (and the distance/window math
    that depends on its sparse output) behind torch.compiler.disable so the
    surrounding model still compiles. The result tensors are normal CUDA tensors
    that flow back into the compiled graph.
    """
    edge_index = _radius_graph(
        pos, r=r_max, batch=batch, loop=True, max_num_neighbors=max_num_neighbors,
    )
    src, dst = edge_index
    dists = (pos[src] - pos[dst]).norm(dim=-1)
    edge_window = _polynomial_cutoff(dists, r_max=r_max, p=p)
    return edge_index, edge_window


class _Sin(nn.Module):
    """Module wrapper for torch.sin so it composes inside nn.Sequential."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


# Module-style activation registry, parallel to the functional one inside __init__.
# Used by the readout heads (which live inside nn.Sequential and need an nn.Module).
_ACTIVATION_MODULES = {
    "gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU, "mish": nn.Mish, "sin": _Sin,
}


class PlatonicTransformer(nn.Module):
    """
    A Transformer architecture equivariant to the symmetries of a specified Platonic solid.

    This model processes point cloud data. It first embeds input node features, then
    "lifts" them into a group-equivariant feature space. A series of PlatonicBlocks
    process these features equivariantly. Finally, for graph-level tasks, it pools
    over the nodes and the group to produce a single invariant prediction. For node-level
    tasks, it pools over the group axis to produce invariant node predictions.

    Args:
        input_dim (int): Dimensionality of the initial node features.
        hidden_dim (int): The per-group-element channel dimension used throughout the model.
        output_dim (int): Dimensionality of the final output.
        nhead (int): Number of attention heads in each PlatonicBlock.
        num_layers (int): Number of PlatonicBlock layers.
        solid_name (str): The name of the Platonic solid ('tetrahedron', 'octahedron',
                          'icosahedron') to define the symmetry group.
        ffn_dim_factor (int): Multiplier for the feed-forward network's hidden dimension,
                              relative to `hidden_dim`.
        scalar_task_level (str): "node" or "graph". Determines the pooling strategy.
        dropout (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0.
        layer_scale_init_value (Optional[float]): Initial value for LayerScale. Default: None.
        **kwargs: Additional keyword arguments for the PlatonicBlock layers
    """
    def __init__(self,
        # Basic/essential specification:
        input_dim: int,
        input_dim_vec: int,
        hidden_dim: int,
        output_dim: int,
        output_dim_vec: int,
        nhead: int,
        num_layers: int,
        solid_name: str,
        spatial_dim: int = 3,
        dense_mode: bool = False, # force dense mode, even if batch is provided
        # Pooling and readout specification:
        scalar_task_level: str = "graph",
        vector_task_level: str = "node",
        ffn_readout: bool = True,
        # Attention block specification:
        mean_aggregation: bool = False,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
        attention: bool = False,
        ffn_dim_factor: int = 4,
        norm_type: str = "layernorm",
        # RoPE and APE specification:
        rope_sigma: float = 1.0,  # if None it is not used
        ape_sigma: float = None,  # if None it is not used
        learned_freqs: bool = True,
        freq_init: str = 'random',
        use_key: bool = False,
        rope_on_values: bool = False,
        attention_backend: str = "scatter",  # "scatter" | "flash"
        qk_norm: bool = False,
        swiglu: bool = False,
        qk_dim_factor: int = 1,
        v_dim_factor: int = 1,
        rope_v_independent: bool = False,
        activation: str = "gelu",
        readout_activation: Optional[str] = None,  # None → fall back to nn.GELU (legacy)
        # Per-block chg/spin conditioning paths (mutually exclusive layerwise vs FiLM):
        chgspin_layerwise: bool = False,
        chgspin_layerwise_gate: bool = False,  # learnable per-block scalar γ_chgspin
        chgspin_film: bool = False,            # per-block FiLM: x → (1+γ_c)·x + β_c
        chgspin_embed_dim: Optional[int] = None,  # per-G-slot chgspin dim = hidden_dim // G
        chgspin_init_gamma: float = 1e-4,         # init for the learnable layerwise gate
        # Radius graph (sparse attention). None → dense within-graph attention
        # (current behavior). When set, computes radius_graph(pos, r=interaction_radius)
        # once in forward and reuses across layers; multiplies softmax-normalized
        # attention weights by a Klicpera polynomial cutoff for a smooth boundary.
        # Requires graph mode (dense_mode=False) and attention_backend='scatter'.
        interaction_radius: Optional[float] = None,
        cutoff_p: int = 6,                  # polynomial exponent for PolynomialCutoff
        max_num_neighbors: int = 1000,      # safety cap for torch_cluster.radius_graph
        # Dual-stage local→global blocks (AllScAIP-style). When True each "layer"
        # expands into TWO PlatonicBlocks (~2× params and FLOPs): the first uses
        # the radius edge_index (local), the second runs full within-graph
        # attention (global). Requires interaction_radius to be set.
        local_global: bool = False,
    ):
        super().__init__()

        # Resolve activation function from string
        _activations = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu, "mish": F.mish, "sin": torch.sin}
        if activation not in _activations:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(_activations.keys())}")
        _activation_fn = _activations[activation]

        # Pick the readout activation Module class. None → nn.GELU (preserves the
        # historical hardcoded default before this option existed).
        if readout_activation is None:
            _readout_act_cls = nn.GELU
        else:
            if readout_activation not in _ACTIVATION_MODULES:
                raise ValueError(
                    f"Unknown readout_activation '{readout_activation}'. "
                    f"Choose from {list(_ACTIVATION_MODULES.keys())} or None."
                )
            _readout_act_cls = _ACTIVATION_MODULES[readout_activation]

        if scalar_task_level not in ["node", "graph"]:
            raise ValueError("scalar_task_level must be 'node' or 'graph'.")

        if vector_task_level not in ["node", "graph"]:
            raise ValueError("vector_task_level must be 'node' or 'graph'.")

        # --- Group and Dimension Setup ---
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        self.hidden_dim = hidden_dim
        self.scalar_task_level = scalar_task_level
        self.vector_task_level = vector_task_level
        self.dense_mode = dense_mode
        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec
        self.mean_aggregation = mean_aggregation

        # --- Radius graph configuration ---
        self.interaction_radius = interaction_radius
        self.cutoff_p = cutoff_p
        self.max_num_neighbors = max_num_neighbors
        self.local_global = local_global
        if interaction_radius is not None:
            if dense_mode:
                raise ValueError(
                    f"interaction_radius={interaction_radius} requires dense_mode=False (graph mode)."
                )
            if attention_backend in ("flash", "flash3") and not local_global:
                raise ValueError(
                    f"interaction_radius={interaction_radius} without local_global requires "
                    "attention_backend='scatter': single-stream radius attention is fully "
                    "sparse and flash/flash3 cannot consume a sparse edge set. With "
                    "local_global=True flash/flash3 is used for the global sub-blocks "
                    "(local sub-blocks force scatter)."
                )
            if not attention:
                raise ValueError("interaction_radius requires attention=True.")
            # _radius_graph is guaranteed non-None (torch_cluster wheel OR
            # pure-PyTorch fallback) so no import check needed here.
        if local_global and interaction_radius is None:
            raise ValueError("local_global=True requires interaction_radius to be set.")

        # Global position embedding for fixed patching ViTs
        if ape_sigma is not None:
            self.ape = APE(hidden_dim, solid_name, ape_sigma, spatial_dim, learned_freqs)
        else:
            self.register_buffer('ape', None)

        # --- Modules ---
        # 1. Input Embedding: Applied before lifting to the group.
        # Maps input features to the per-group-element hidden dimension.
        self.x_embedder = PlatonicLinear((input_dim + input_dim_vec * spatial_dim) * self.num_G, self.hidden_dim, solid_name, bias=False)

        # 2. Equivariant Encoder Layers
        # The blocks operate on the total flattened dimension (G * C).
        dim_feedforward = int(self.hidden_dim * ffn_dim_factor)

        # In local_global mode, each "logical layer" expands into two PlatonicBlocks
        # (local at even index, then global at odd index). num_blocks == num_layers
        # in normal mode and 2*num_layers in dual-stage mode. The forward loop uses
        # the even/odd index to decide which blocks receive edge_index/window.
        #
        # Per-block attention backend: local sub-blocks must use scatter (they
        # consume the sparse radius edge set, which flash cannot). Global
        # sub-blocks run full within-graph attention, so they use the configured
        # attention_backend — flash stays available there. In single-stream mode
        # every block uses attention_backend (validation forces scatter above when
        # interaction_radius is set).
        self.num_blocks = num_layers * (2 if local_global else 1)
        self.layers = nn.ModuleList()
        for i in range(self.num_blocks):
            if local_global and (i % 2 == 0):
                block_backend = "scatter"
            else:
                block_backend = attention_backend
            self.layers.append(PlatonicBlock(
                d_model=self.hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                solid_name=solid_name,
                dropout=dropout,
                activation=_activation_fn,
                drop_path=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                norm_type=norm_type,
                freq_sigma=rope_sigma,
                freq_init=freq_init,
                learned_freqs=learned_freqs,
                spatial_dims=spatial_dim,
                mean_aggregation=mean_aggregation,
                attention=attention,
                use_key=use_key,
                rope_on_values=rope_on_values,
                attention_backend=block_backend,
                qk_norm=qk_norm,
                swiglu=swiglu,
                qk_dim_factor=qk_dim_factor,
                v_dim_factor=v_dim_factor,
                rope_v_independent=rope_v_independent,
            ))

        if ffn_readout:
            self.scalar_readout = nn.Sequential(
                PlatonicLinear(self.hidden_dim, self.hidden_dim, solid_name),
                _readout_act_cls(),
                PlatonicLinear(self.hidden_dim, self.num_G * output_dim, solid_name)
            )
            self.vector_readout = nn.Sequential(
                PlatonicLinear(self.hidden_dim, self.hidden_dim, solid_name),
                _readout_act_cls(),
                PlatonicLinear(self.hidden_dim, self.num_G * output_dim_vec * spatial_dim, solid_name)
            )
        else:
            self.scalar_readout = PlatonicLinear(self.hidden_dim, self.num_G * output_dim, solid_name)
            self.vector_readout = PlatonicLinear(self.hidden_dim, self.num_G * output_dim_vec * spatial_dim, solid_name)

        # --- Per-block chg/spin conditioning ---
        self.chgspin_layerwise = chgspin_layerwise
        self.chgspin_layerwise_gate = chgspin_layerwise_gate
        self.chgspin_film = chgspin_film
        if chgspin_layerwise and chgspin_film:
            raise ValueError("chgspin_layerwise and chgspin_film are mutually exclusive.")
        if chgspin_layerwise_gate and not chgspin_layerwise:
            raise ValueError("chgspin_layerwise_gate requires chgspin_layerwise=True.")
        if chgspin_film and chgspin_embed_dim is None:
            raise ValueError("chgspin_film=True requires chgspin_embed_dim.")

        # Learnable per-block scalar for the additive injection. Initialised to
        # chgspin_init_gamma (typically layer_scale_init_value) so the per-block
        # add starts at a magnitude comparable to one block's attention contribution.
        # In local_global mode, each sub-block (local and global) gets its own gate.
        if chgspin_layerwise_gate:
            self.gamma_chgspin = nn.Parameter(
                chgspin_init_gamma * torch.ones(self.num_blocks)
            )

        # FiLM: per-block Linear(embed_dim → 2*embed_dim) producing (γ_c, β_c).
        # Zero-init weights and bias so FiLM is the identity at start; the model
        # learns the modulation through gradient descent. One FiLM proj per
        # PlatonicBlock (= 2 × num_layers in local_global mode).
        if chgspin_film:
            self.film_projs = nn.ModuleList([
                nn.Linear(chgspin_embed_dim, 2 * chgspin_embed_dim)
                for _ in range(self.num_blocks)
            ])
            for proj in self.film_projs:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

    def _lift_invariant(self, token: Tensor) -> Tensor:
        """Replicate a per-G-slot signal (..., embed_dim) across G to (..., G*embed_dim).
        Group-invariant lift: identical value in every G slot, so it commutes with the
        group action that permutes the G axis (matches the PlatonicBlock storage layout).
        """
        embed_dim = token.shape[-1]
        expanded = list(token.shape[:-1]) + [self.num_G, embed_dim]
        return token.unsqueeze(-2).expand(expanded).reshape(
            *token.shape[:-1], self.num_G * embed_dim
        )

    def forward(self,
                x: Tensor,
                pos: Tensor,
                batch: Optional[torch.Tensor] = None,
                mask: Optional[Tensor] = None,
                vec: Optional[Tensor] = None,
                avg_num_nodes: float = 1.0,
                chgspin_mixed_per_node: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the Platonic Transformer.

        Args:
            x (Tensor): Input node features of shape (N, input_dim).
            pos (Tensor): Node positions of shape (N, spatial_dims).
            batch (Tensor): Batch index for each node of shape (N,).
            mask (Tensor, optional): Attention mask of shape (B, N) or (N, N) for dense inputs.
            chgspin_mixed_per_node (Tensor, optional): Per-node chg/spin conditioning
                in *per-G-slot* space, shape (N, embed_dim) in graph mode (or
                (B, N, embed_dim) if input is already dense). Drives the per-block
                additive injection (chgspin_layerwise) or FiLM modulation
                (chgspin_film) — lifted to hidden_dim internally.
        Returns:
            Tensor: Final predictions. Shape is (B, output_dim) for graph tasks
                    or (N, output_dim) for node tasks.
        """

        # 1. Convert to dense format if needed
        if self.dense_mode:
            self._input_was_dense_format = (batch is None)
            # Densify the per-node chg/spin signal alongside x so per-block
            # injection lines up with the (B, N_max, ...) layout.
            if chgspin_mixed_per_node is not None and not self._input_was_dense_format:
                chgspin_mixed_per_node, _, _, _ = to_dense_and_mask(
                    chgspin_mixed_per_node, None, pos, batch
                )
            x, vec, pos, mask = to_dense_and_mask(x, vec, pos, batch)
            batch = None
        else:
            self._input_was_dense_format = False
            mask = None

        # 2. Lift scalars and vectors, then embed
        x = lift(x, vec, self.group)
        x = self.x_embedder(x)  # [..., N, num_patches * C]
        x = x + self.ape(pos) if self.ape is not None else x  # Add absolute position embedding

        # Pre-loop: lift the additive chg/spin token once (no per-block projection
        # for the layerwise path, so the same lifted token is reused every block).
        chgspin_token = None
        if self.chgspin_layerwise and chgspin_mixed_per_node is not None:
            chgspin_token = self._lift_invariant(chgspin_mixed_per_node)

        # Radius graph + polynomial cutoff computed once and reused across all blocks.
        # Positions are constant through the layers in the scatter path, so the edge
        # set is shared. With loop=True every node has at least its self-loop
        # (window=1 at dist=0) so outputs are always non-zero. The call is isolated
        # behind torch.compiler.disable because torch_cluster.radius has no
        # fake-tensor implementation; Dynamo would otherwise fail to trace it.
        edge_index = None
        edge_window = None
        if self.interaction_radius is not None and not self.dense_mode and batch is not None:
            edge_index, edge_window = _build_radius_edges(
                pos, batch,
                r_max=self.interaction_radius,
                p=self.cutoff_p,
                max_num_neighbors=self.max_num_neighbors,
            )

        # 3. Equivariant Encoder (Platonic Conv Blocks)
        for i, layer in enumerate(self.layers):
            if self.chgspin_film and chgspin_mixed_per_node is not None:
                # Per-block FiLM: x → (1 + γ_c) * x + β_c, where γ_c, β_c are
                # produced by a zero-init Linear(embed_dim → 2*embed_dim) on the
                # per-node chg/spin signal, then lifted group-invariantly across G.
                film_out = self.film_projs[i](chgspin_mixed_per_node)
                gamma_c, beta_c = film_out.chunk(2, dim=-1)
                gamma_lifted = self._lift_invariant(gamma_c)
                beta_lifted = self._lift_invariant(beta_c)
                x = (1 + gamma_lifted) * x + beta_lifted
            elif chgspin_token is not None:
                # Layerwise additive injection, optionally gated.
                if self.chgspin_layerwise_gate:
                    x = x + self.gamma_chgspin[i] * chgspin_token
                else:
                    x = x + chgspin_token
            # Route edge_index/edge_window only to local sub-blocks. In normal
            # mode every block is "local" (gets the full edge set, or None when
            # interaction_radius is not configured). In local_global mode, even-
            # indexed blocks are local and odd-indexed are global (full attention).
            if self.local_global and (i % 2 == 1):
                block_edge_index = None
                block_edge_window = None
            else:
                block_edge_index = edge_index
                block_edge_window = edge_window

            x = layer(
                x=x,
                pos=pos,
                batch=batch,
                mask=mask,
                avg_num_nodes=avg_num_nodes,
                edge_index=block_edge_index,
                edge_window=block_edge_window,
            )

        # 4. Post-pooling readout
        if self.scalar_task_level == "graph":
            scalar_x = pool(x, batch, mask, avg_num_nodes, self.dense_mode, self.mean_aggregation)
        else:
            if not self._input_was_dense_format and self.dense_mode:
                scalar_x = x[mask]
            else:
                scalar_x = x

        if self.vector_task_level == "graph":
            vector_x = pool(x, batch, mask, avg_num_nodes, self.dense_mode, self.mean_aggregation)
        else:
            if not self._input_was_dense_format and self.dense_mode:
                vector_x = x[mask]
            else:
                vector_x = x

        # Readout precision: match the readout weights' dtype rather than hard-casting
        # to fp64. Under the default fp32 path this is a no-op; under a +precision=fp64
        # preset that casts model parameters to fp64, activations are upcast accordingly
        # so the matmul runs at 53-bit. TF32 is also disabled around the readout so the
        # fp32 path gets a true 23-bit mantissa rather than ~10-bit TF32 accumulator.
        readout_dtype = next(self.scalar_readout.parameters()).dtype
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            scalar_x = self.scalar_readout(scalar_x.to(readout_dtype))
            vector_x = self.vector_readout(vector_x.to(readout_dtype))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

        # 5. Extract the scalar and vector parts
        scalars = to_scalars_vectors(scalar_x, self.output_dim, 0, self.group)[0]
        vectors = to_scalars_vectors(vector_x, 0, self.output_dim_vec, self.group)[1]

        # Return final result
        return scalars, vectors
