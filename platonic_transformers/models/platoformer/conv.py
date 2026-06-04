import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# Native PyTorch scatter ops — torch.compile compatible, no torch_scatter dependency.
# Both scatter_sum and scatter_softmax promote to fp32 internally and cast back to
# the input dtype on return. Under bf16-mixed autocast, summing many bf16 values
# in scatter_add_ accumulates in bf16 (no fp32 accumulator promotion for scatter
# ops in PyTorch), and computing softmax math entirely in bf16 compounds ~1% per
# exp + ~sqrt(N) per neighbor reduction → multi-percent error in attention weights
# per layer. Flash attention masks this with internal fp32 accumulators; this
# scatter implementation must do the same explicitly. The fp32 promotion is a no-op
# under fp32 callers (.float() on fp32 = identity, autocast wrapper inert outside
# autocast scope).
def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 dim_size: int = None) -> torch.Tensor:
    """Native PyTorch replacement for torch_scatter.scatter_sum (fp32-internal)."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    orig_dtype = src.dtype
    with torch.amp.autocast(device_type=src.device.type, enabled=False):
        src_fp32 = src.float()
        out_shape = list(src_fp32.shape)
        out_shape[dim] = dim_size
        out = torch.zeros(out_shape, dtype=torch.float32, device=src.device)
        idx = index
        while idx.dim() < src_fp32.dim():
            idx = idx.unsqueeze(-1)
        idx = idx.expand_as(src_fp32)
        result = out.scatter_add_(dim, idx, src_fp32)
    return result.to(orig_dtype)


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                     dim_size: int = None) -> torch.Tensor:
    """Native PyTorch replacement for torch_scatter.scatter_softmax (fp32-internal)."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    orig_dtype = src.dtype
    with torch.amp.autocast(device_type=src.device.type, enabled=False):
        src_fp32 = src.float()
        max_vals = torch.zeros(dim_size, dtype=torch.float32, device=src.device)
        max_vals.scatter_reduce_(0, index, src_fp32, reduce="amax", include_self=False)
        src_shifted = src_fp32 - max_vals[index]
        exp_src = torch.exp(src_shifted)
        sum_exp = torch.zeros(dim_size, dtype=torch.float32, device=src.device)
        sum_exp.scatter_add_(0, index, exp_src)
        result = exp_src / (sum_exp[index] + 1e-30)
    return result.to(orig_dtype)

try:
    from torch_cluster import knn_graph
except (ImportError, OSError):
    knn_graph = None

try:
    from torch_cluster import radius_graph
except (ImportError, OSError):
    radius_graph = None

try:
    from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]
except (ImportError, OSError):  # flash-attn is optional
    flash_attn_varlen_func = None

# Flash-Attention 3 (Hopper-only, sm_90a). Built from the flash-attention
# hopper/ subdir and installed as a separate package (flash_attn_3). Coexists
# with FA2. Key advantages: WGMMA + TMA kernels (~1.5–2× faster than FA2 on
# H100) and native support for head_dim_q != head_dim_v in the varlen API.
try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_3  # type: ignore[import-not-found]
except (ImportError, OSError):
    flash_attn_varlen_func_3 = None

from .utils import scatter_add
from .rope import PlatonicRoPE
from .linear import PlatonicLinear
from .groups import PLATONIC_GROUPS
from .io import lift_vectors


class PlatonicConv(nn.Module):
    """
    Computes a group-equivariant dynamic convolution supporting both graph and dense modes.

    This layer uses Rotary Positional Embeddings (RoPE) to compute a dynamic
    convolution kernel. It supports two modes for dense data:
    1.  attention=False (Default): A highly efficient linear convolutio type "attention" mechanism.
    2.  attention=True: Equivariant scaled dot-product attention with softmax.
    
    Graph-structured data only uses the linear attention mechanism.
    The layer is equivariant to the symmetries of a specified Platonic solid.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        solid_name: str,
        spatial_dims: int = 3,
        freq_sigma: float = 1.0,
        freq_init: str = 'random',
        learned_freqs: bool = True,
        bias: bool = True,
        mean_aggregation: bool = False,
        attention: bool = False,
        use_key: bool = False,
        rope_on_values: bool = False,
        attention_backend: str = "scatter",
        qk_norm: bool = False,
        qk_dim_factor: int = 1,
        v_dim_factor: int = 1,
        rope_v_independent: bool = False,
    ):
        super().__init__()

        # --- Group Setup ---
        self.rope_on_values = rope_on_values
        if attention_backend not in ("scatter", "flash", "flash3"):
            raise ValueError(
                f"attention_backend must be 'scatter', 'flash', or 'flash3', got {attention_backend!r}"
            )
        if attention_backend == "flash" and flash_attn_varlen_func is None:
            raise ImportError(
                "attention_backend='flash' requires the flash-attn package "
                "(pip install flash-attn)."
            )
        if attention_backend == "flash3" and flash_attn_varlen_func_3 is None:
            raise ImportError(
                "attention_backend='flash3' requires flash-attn 3 (Hopper-only). "
                "Build from flash-attention/hopper/ and install (package name 'flash_attn_3')."
            )
        self.attention_backend = attention_backend
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        
        # --- Dimension Validation and Setup ---
        if in_channels % self.num_G != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by group size ({self.num_G}).")
        self.in_channels_g = in_channels // self.num_G

        if out_channels % self.num_G != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by group size ({self.num_G}).")
        self.out_channels_g = out_channels // self.num_G

        if num_heads % self.num_G != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by group size ({self.num_G}).")

        if embed_dim % (num_heads // self.num_G) != 0:
             raise ValueError(f"embed_dim ({embed_dim}) must be divisible by (num_heads // num_group) = "
                             f"{self.num_G // num_heads}.")
        self.embed_dim = embed_dim
        self.embed_dim_g = embed_dim // self.num_G

        self.out_channels = out_channels
        self.effective_num_heads = num_heads//self.num_G
        self.head_dim = self.embed_dim_g // self.effective_num_heads

        # Asymmetric Q/K vs V dim. qk_dim_factor expands Q and K head_dim by an
        # integer factor; v_dim_factor independently expands V (and the output
        # projection input). Common settings:
        #   qkdf=2, vdf=1: Q/K wider for more RoPE frequencies; V padded with
        #     zeros to match for flash-attn, output truncated. Same V capacity
        #     as baseline. (V-padding hack — wastes flash kernel work on zeros
        #     but keeps V/output projections small.)
        #   qkdf=2, vdf=2: full expansion. No padding; V also wider. More V
        #     capacity but larger V/output projections.
        #   qkdf=1, vdf=1: baseline (no change).
        # Requires v_dim_factor <= qk_dim_factor (asymmetric attention with
        # V wider than Q/K is not supported).
        self.qk_dim_factor = int(qk_dim_factor)
        self.v_dim_factor = int(v_dim_factor)
        if self.qk_dim_factor < 1 or self.v_dim_factor < 1:
            raise ValueError(
                f"qk_dim_factor and v_dim_factor must be >= 1, got "
                f"qk_dim_factor={qk_dim_factor}, v_dim_factor={v_dim_factor}."
            )
        if self.v_dim_factor > self.qk_dim_factor:
            raise ValueError(
                f"v_dim_factor ({self.v_dim_factor}) must be <= qk_dim_factor "
                f"({self.qk_dim_factor})."
            )
        self.embed_dim_qk = embed_dim * self.qk_dim_factor
        self.head_dim_qk = self.head_dim * self.qk_dim_factor
        self.embed_dim_v = embed_dim * self.v_dim_factor
        self.head_dim_v = self.head_dim * self.v_dim_factor
        if self.qk_dim_factor > 1 or self.v_dim_factor > 1:
            if attention_backend not in ("flash", "flash3"):
                raise ValueError(
                    "qk_dim_factor>1 or v_dim_factor>1 currently requires "
                    "attention_backend='flash' or 'flash3' (scatter/SDPA paths not yet plumbed)."
                )
            if not attention:
                raise ValueError("qk_dim_factor>1 or v_dim_factor>1 requires attention=True.")

        self.mean_aggregation = mean_aggregation
        self.attention = attention
        self.rope_on_values = rope_on_values

        # --- Sub-modules ---
        self.use_key = use_key
        self.q_proj = PlatonicLinear(in_channels, self.embed_dim_qk, solid_name, bias=bias)
        self.v_proj = PlatonicLinear(in_channels, self.embed_dim_v, solid_name, bias=bias)
        if freq_sigma is None or use_key:
            self.k_proj = PlatonicLinear(in_channels, self.embed_dim_qk, solid_name, bias=bias)
        else:
            self.register_buffer('k_proj', None)

        # Group-equivariant RoPE for Q/K — sized to head_dim_qk (the expanded width).
        # V-side RoPE (only when qk_dim_factor>1 AND rope_on_values=True) is a
        # separate module at head_dim. At qk_dim_factor=1, rope_emb_v is None and
        # the forward path falls back to rope_emb for V, recovering the original
        # behavior exactly.
        if freq_sigma is not None:
            self.rope_emb = PlatonicRoPE(
                embed_dim=self.embed_dim_qk,
                num_heads=self.effective_num_heads,
                head_dim=self.head_dim_qk,
                solid_name=solid_name,
                spatial_dims=spatial_dims,
                freq_sigma=freq_sigma,
                learned_freqs=learned_freqs,
                freq_init=freq_init
            )
            # V-side RoPE: a separate module is created when EITHER (a) V's
            # head_dim differs from Q/K's (shape mismatch, required), OR
            # (b) rope_v_independent=True (decouple V's RoPE frequencies from
            # Q/K's even at matching shape). Independent V freqs at no channel
            # cost — doubles the effective spatial-direction coverage used in
            # attention (QK's freqs determine "which atoms to attend to";
            # V's freqs determine "how the retrieved features encode position").
            # Equivariance: each PlatonicRoPE is internally group-equivariant;
            # and the inverse rotation on the attention output uses the SAME
            # rope_emb_v module that was applied forward to V (see
            # _forward_graph below), so ρ(g_i)⁻¹ on the output cancels exactly
            # the ρ(g_j) that rotated v_j — GTA Eq. 5 holds.
            need_separate_v_rope = self.rope_on_values and (
                self.head_dim_v != self.head_dim_qk or rope_v_independent
            )
            if need_separate_v_rope:
                self.rope_emb_v = PlatonicRoPE(
                    embed_dim=self.embed_dim_v,
                    num_heads=self.effective_num_heads,
                    head_dim=self.head_dim_v,
                    solid_name=solid_name,
                    spatial_dims=spatial_dims,
                    freq_sigma=freq_sigma,
                    learned_freqs=learned_freqs,
                    freq_init=freq_init
                )
            else:
                self.rope_emb_v = None
        else:
            self.register_buffer('rope_emb', None)
            self.rope_emb_v = None

        # Optional QK-norm: RMSNorm on Q and K independently, applied after
        # projection (and before RoPE) — LLaMA-3 style. The norm operates on
        # the (expanded) head_dim_qk axis with γ of shape (head_dim_qk,), shared
        # across the G axis so the operation commutes with the Platonic group
        # action.
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim_qk)
            self.k_norm = nn.RMSNorm(self.head_dim_qk)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Final equivariant linear layer. Input dim follows V (= embed_dim_v),
        # which grows with v_dim_factor; out_channels unchanged.
        self.out_proj = PlatonicLinear(self.embed_dim_v, out_channels, solid_name, bias=bias)

    def _forward_shared(self, x: Tensor, pos: Tensor):
        """Shared logic for projections and RoPE application."""
        leading_dims = x.shape[:-1]
        
        q_raw = self.q_proj(x)
        v_raw = self.v_proj(x)
        # If not using RoPE, then project, but if using RoPE then use ones.
        # When use_key=False (and RoPE is on), K is ones at the QK-expanded
        # width so RoPE-rotation is the only Q-K signal.
        k_raw = self.k_proj(x) if ((self.rope_emb is None) or self.use_key) else torch.ones_like(q_raw)

        # Reshape for multi-head processing. Q/K use head_dim_qk; V uses
        # head_dim_v (= head_dim when v_dim_factor=1).
        q = q_raw.view(*leading_dims, self.num_G, self.effective_num_heads, self.head_dim_qk)
        k = k_raw.view(*leading_dims, self.num_G, self.effective_num_heads, self.head_dim_qk)
        v = v_raw.view(*leading_dims, self.num_G, self.effective_num_heads, self.head_dim_v)

        # QK-norm (RMSNorm on head_dim_qk) — applied before RoPE so rotation
        # acts on already-normalized vectors. Identity when qk_norm=False.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to query and key (and optionally value, per GTA Eq. 5).
        # Q/K use rope_emb (sized to head_dim_qk); V uses rope_emb_v when
        # qk_dim_factor>1 (separate module at head_dim), else rope_emb.
        if self.rope_emb is not None:
            q = self.rope_emb(q, pos)
            k = self.rope_emb(k, pos)
            if self.rope_on_values:
                rope_v = self.rope_emb_v if self.rope_emb_v is not None else self.rope_emb
                v = rope_v(v, pos)

        return q, k, v

    def graph_scattered_attention(self,
        q: torch.Tensor,      # [N, G, H, D]
        k: torch.Tensor,      # [N, G, H, D]
        v: torch.Tensor,      # [N, G, H, D]
        batch: torch.Tensor,  # [N]
        pos: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        k_knn: Optional[int] = None,
        edge_window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute fully connected edges if edge_index is None, or kNN edges if k_knn is given.
        Uses an equivariant attention mechanism over the combined group/head dimension.

        If `edge_window` is provided (shape [E]), it multiplies the softmax-normalized
        attention weights per edge before the scatter-aggregation. Used by the
        radius-graph path with a polynomial cutoff to smoothly attenuate boundary
        atoms — matches the cutoff-on-kernel-basis convention from ponita.

        Returns
        -------
        out : Tensor, shape [N, G*H*D]
        """
        if self.qk_dim_factor > 1:
            raise NotImplementedError(
                "qk_dim_factor>1 is not yet supported for the scatter attention path; "
                "use attention_backend='flash'."
            )
        N, G, H, D = q.shape
        device = q.device

        if edge_index is not None:
            src, dst = edge_index.to(device)
        elif k_knn is not None:
            if pos is None:
                raise ValueError("k_knn was given but 'pos' is None.")
            if knn_graph is None:
                raise ImportError("torch_cluster.knn_graph is required for kNN mode.")
            edge_index = knn_graph(
                x=pos.to(device), k=k_knn, batch=batch, loop=True
            )                                        # [2, |E_knn|]
            src, dst = edge_index
        else:
            N = batch.shape[0]
            # Create a dense N x N grid of all possible edges
            node_idx = torch.arange(N, device=device)
            src, dst = torch.meshgrid(node_idx, node_idx, indexing='ij')

            # Keep only the edges where the source and destination nodes
            # belong to the same graph in the batch.
            mask = batch[src] == batch[dst]
            
            # Apply the mask to get the final edge index
            src = src[mask]
            dst = dst[mask]

        E = src.numel()
        
        GH = G * H
        q_src = q.reshape(N, GH, D)[src]  # [E, GH, D]
        k_dst = k.reshape(N, GH, D)[dst]  # [E, GH, D]
        v_dst = v.reshape(N, GH, D)[dst]  # [E, GH, D]

        scores = (q_src * k_dst).sum(-1) * D ** -0.5  # [E, GH]

        # reindex ids for heads
        head_ids = torch.arange(GH, device=device).repeat(E, 1)     # [E, GH]
        group_ids = src.unsqueeze(1) * GH + head_ids                # [E, GH]

        a = _scatter_softmax(
            scores.flatten(),
            group_ids.flatten(),
            dim=0,
            dim_size=N * GH
        ).view(E, GH)

        # Polynomial cutoff: attenuate per-edge contribution smoothly to 0 at r_max.
        # The softmax sum-to-1 invariant is intentionally broken here — boundary
        # atoms simply contribute less. The self-loop (dist=0, window=1) guarantees
        # every node still has a non-zero output.
        if edge_window is not None:
            a = a * edge_window.unsqueeze(-1)

        weighted = (a.unsqueeze(-1) * v_dst).reshape(-1, D)         # [E*GH, D]

        out = _scatter_sum(
            weighted,
            group_ids.flatten(),
            dim=0,
            dim_size=N * GH
        ).view(N, GH, D)

        # Reshape (N, GH, D) -> (N, G*H*D)
        return out.reshape(N, G * H * D)
        


    def graph_flash_varlen_attention(self,
        q: torch.Tensor,      # [N, G, H, D]
        k: torch.Tensor,      # [N, G, H, D]
        v: torch.Tensor,      # [N, G, H, D]
        batch: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """Equivalent to graph_scattered_attention (fully-connected within-graph)
        but computed via flash_attn_varlen_func — one fused CUDA kernel, no
        explicit N x N meshgrid/mask.

        Parameters
        ----------
        q, k, v : [N, G, H, D]  query, key, value tensors (post-RoPE).
        batch   : [N]  per-node graph index (values 0..B-1, sorted by graph).

        Returns
        -------
        out : [N, G*H*D]

        Equivariance: the G axis is treated as an extra head dimension.
        Because the group acts on the G axis by permutation, and flash varlen
        applies independent attention per head, the operation commutes with
        the group action -> equivariance preserved (matches the scatter
        implementation semantics).

        Dtype: flash_attn_varlen_func requires fp16/bf16. We cast q,k,v to
        bf16 inside the kernel call and cast the output back to the input
        dtype. For fp32 callers this means a ~bf16 level of rounding error
        inside attention but the residual path remains fp32.
        """
        # q,k: head_dim D_qk;  v: head_dim D_v (= D_qk when qk_dim_factor=1).
        N, G, H, D_qk = q.shape
        D_v = v.shape[-1]
        GH = G * H
        device = q.device

        # Contiguous flattened views.
        q_flat = q.reshape(N, GH, D_qk).contiguous()
        k_flat = k.reshape(N, GH, D_qk).contiguous()
        # Pad V to D_qk so flash-attn 2.x can fuse (it requires matching head
        # dims across Q/K/V). The padded channels are zero, so attention's
        # output in those channels is also zero — truncating after the call
        # is lossless. Trade-off: ~qk_dim_factor× extra V-load/V-output work
        # inside the kernel.
        if D_v != D_qk:
            v_padded = F.pad(v, (0, D_qk - D_v))
        else:
            v_padded = v
        v_flat = v_padded.reshape(N, GH, D_qk).contiguous()

        # Cumulative sequence lengths [B+1] and max seq length in this batch.
        # We assume `batch` is sorted (standard PyG convention), so counts can
        # be derived by one bincount pass.
        B = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=B).to(dtype=torch.int32)
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = torch.cumsum(counts, dim=0)
        max_seqlen = int(counts.max().item())

        # Flash Attention requires fp16 / bf16. Cast before call, cast back after.
        orig_dtype = q_flat.dtype
        q_bf = q_flat.to(torch.bfloat16)
        k_bf = k_flat.to(torch.bfloat16)
        v_bf = v_flat.to(torch.bfloat16)

        out_bf = flash_attn_varlen_func(
            q_bf, k_bf, v_bf,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )  # [N, GH, D_qk]

        out = out_bf.to(orig_dtype)
        # Truncate the padded V channels back to D_v.
        if D_v != D_qk:
            out = out[..., :D_v].contiguous()
        return out.reshape(N, GH * D_v)

    def graph_flash3_varlen_attention(self,
        q: torch.Tensor,      # [N, G, H, D_qk]
        k: torch.Tensor,      # [N, G, H, D_qk]
        v: torch.Tensor,      # [N, G, H, D_v]
        batch: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """Hopper-native FA3 varlen attention. Like graph_flash_varlen_attention
        but uses the flash-attn 3 kernel: WGMMA + TMA on sm_90a, ~1.5–2× faster
        than FA2 on H100, AND natively supports head_dim_qk != head_dim_v
        (so the V-padding hack is no longer needed when qk_dim_factor != v_dim_factor).
        """
        N, G, H, D_qk = q.shape
        D_v = v.shape[-1]
        GH = G * H
        device = q.device

        q_flat = q.reshape(N, GH, D_qk).contiguous()
        k_flat = k.reshape(N, GH, D_qk).contiguous()
        v_flat = v.reshape(N, GH, D_v).contiguous()

        B = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=B).to(dtype=torch.int32)
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = torch.cumsum(counts, dim=0)
        max_seqlen = int(counts.max().item())

        orig_dtype = q_flat.dtype
        q_bf = q_flat.to(torch.bfloat16)
        k_bf = k_flat.to(torch.bfloat16)
        v_bf = v_flat.to(torch.bfloat16)

        out_bf = flash_attn_varlen_func_3(
            q_bf, k_bf, v_bf,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )
        # FA3 may return either just `out` or a tuple (out, softmax_lse, ...) —
        # depending on flags/version. Normalize.
        if isinstance(out_bf, (tuple, list)):
            out_bf = out_bf[0]

        out = out_bf.to(orig_dtype)
        return out.reshape(N, GH * D_v)

    def _forward_graph(self, x: Tensor, pos: Tensor, batch: Tensor, avg_num_nodes=1.0,
                       edge_index: Optional[Tensor] = None,
                       edge_window: Optional[Tensor] = None):
        """
        Implementation for graph-structured data.
        Supports both kernelized linear attention and standard softmax attention.

        If `edge_index` is provided (radius-graph path), the scatter attention uses
        the given edges and (optionally) multiplies per-edge weights by `edge_window`.
        Flash attention does not support sparse edge sets — caller must guarantee
        scatter backend when edge_index is given.
        """
        q_rope, k_rope, v = self._forward_shared(x, pos) # [N, G, H, D_h]

        if self.attention:
            if self.attention_backend in ("flash", "flash3"):
                if edge_index is not None:
                    raise ValueError(
                        f"attention_backend={self.attention_backend!r} does not support "
                        "sparse edge_index. Use attention_backend='scatter' with "
                        "interaction_radius."
                    )
                if self.attention_backend == "flash3":
                    output = self.graph_flash3_varlen_attention(q_rope, k_rope, v, batch)
                else:
                    output = self.graph_flash_varlen_attention(q_rope, k_rope, v, batch)
            else:
                output = self.graph_scattered_attention(
                    q_rope, k_rope, v, batch, pos,
                    edge_index=edge_index, edge_window=edge_window,
                )
            # Un-rotate output (GTA Eq. 5: O_i = ρ(g_i)⁻¹ * weighted_sum).
            # Output is at V's head_dim_v, so use rope_emb_v when QK and V have
            # different head_dims; else fall back to rope_emb.
            if self.rope_on_values and self.rope_emb is not None:
                rope_v = self.rope_emb_v if self.rope_emb_v is not None else self.rope_emb
                output_ghd = output.view(-1, self.num_G, self.effective_num_heads, self.head_dim_v)
                output_ghd = rope_v(output_ghd, pos, inverse=True)
                output = output_ghd.flatten(-3, -1)
        else:
            if self.qk_dim_factor > 1:
                raise NotImplementedError(
                    "qk_dim_factor>1 is not yet supported for the linear (attention=False) path."
                )
            kv_outer_product = torch.einsum('nghd,nghe->nghde', k_rope, v)
            num_graphs = batch.max() + 1
            kv_kernel = scatter_add(kv_outer_product, batch, dim_size=num_graphs)

            if self.mean_aggregation:
                num_nodes = scatter_add(torch.ones_like(batch, dtype=torch.float), batch, dim_size=num_graphs)[..., None, None, None, None]
            else:
                num_nodes = avg_num_nodes
            kv_kernel = kv_kernel / num_nodes

            output = torch.einsum('nghd,nghde->nghe', q_rope, kv_kernel[batch])
            # Un-rotate output for linear attention. Use the V-side RoPE module
            # (rope_emb_v when set, else rope_emb) — must match the forward
            # rotation applied to V in _forward_shared. With rope_v_independent
            # this is rope_emb_v's independent freqs; equivariance preserved.
            if self.rope_on_values and self.rope_emb is not None:
                rope_v = self.rope_emb_v if self.rope_emb_v is not None else self.rope_emb
                output = rope_v(output, pos, inverse=True)
            output = output.flatten(-3, -1) # -> (..., G, H, H_dim) -> (..., G*H*H_dim)

        return self.out_proj(output)

    def _forward_dense(self, x: Tensor, pos: Tensor, mask: Tensor, avg_num_nodes=1.0):
        """
        Implementation for dense, padded data.
        Supports both linear and standard softmax attention.
        """
        if self.qk_dim_factor > 1:
            raise NotImplementedError(
                "qk_dim_factor>1 is not yet supported for the dense (non-graph) path."
            )
        q_rope, k_rope, v = self._forward_shared(x, pos)
        B, S, _ = x.shape # B: batch size, S: sequence length

        if self.attention:
            # Reshape for scaled_dot_product_attention: (B, S, G, H, Dh) -> (B, G*H, S, Dh)
            q_sdpa = q_rope.view(B, S, self.num_G * self.effective_num_heads, self.head_dim).transpose(1, 2)
            k_sdpa = k_rope.view(B, S, self.num_G * self.effective_num_heads, self.head_dim).transpose(1, 2)
            v_sdpa = v.view(B, S, self.num_G * self.effective_num_heads, self.head_dim).transpose(1, 2)

            attn_mask = mask[:, None, None, :] if mask is not None else None
            # Let PyTorch auto-select the fastest SDPA backend (CuDNN on H100, Flash-2 elsewhere)
            attn_output = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask)

            # Reshape back: (B, G*H, S, Dh) -> (B, S, G, H, Dh)
            attn_output = attn_output.transpose(1, 2).view(B, S, self.num_G, self.effective_num_heads, self.head_dim)
            # Un-rotate output (GTA Eq. 5: O_i = ρ(g_i)⁻¹ * weighted_sum).
            # Use the V-side RoPE module (matches forward V rotation) — same
            # selection as the graph-mode flash path.
            if self.rope_on_values and self.rope_emb is not None:
                rope_v = self.rope_emb_v if self.rope_emb_v is not None else self.rope_emb
                attn_output = rope_v(attn_output, pos, inverse=True)
            output = attn_output.reshape(B, S, self.embed_dim)
        else:
            if mask is not None:
                # Apply mask before aggregation
                v = v * mask[..., None, None, None]
                k_rope = k_rope * mask[..., None, None, None]

            kv_kernel = torch.einsum('bsghd,bsghe->bghde', k_rope, v)

            if self.mean_aggregation and mask is not None:
                num_nodes = mask.sum(dim=-1).float().view(B, 1, 1, 1, 1)
            else:
                num_nodes = avg_num_nodes
            kv_kernel = kv_kernel / num_nodes

            output = torch.einsum('bsghd,bghde->bsghe', q_rope, kv_kernel)
            output = output.flatten(-3, -1)
        
        return self.out_proj(output)
    
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        avg_num_nodes: Optional[float] = 1.0,
        edge_index: Optional[Tensor] = None,
        edge_window: Optional[Tensor] = None,
    ) -> Tensor:
        is_graph_mode = batch is not None
        avg_num_nodes = avg_num_nodes if avg_num_nodes is not None else 1.0

        if is_graph_mode:
            if mask is not None:
                raise ValueError("Only one of 'batch' or 'mask' can be provided.")
            return self._forward_graph(
                x, pos, batch, avg_num_nodes=avg_num_nodes,
                edge_index=edge_index, edge_window=edge_window,
            )
        else:
            if edge_index is not None:
                raise ValueError("edge_index is only supported in graph mode (batch != None).")
            return self._forward_dense(x, pos, mask, avg_num_nodes=avg_num_nodes)


class PlatonicEdgeConv(nn.Module):
    """
    Group-equivariant EdgeConv-style patchification.

    For each FPS center i and its k nearest dense points j in KNN(i):
      - compute the relative offset p_j - p_i
      - lift it to G-equivariant features
      - concat per-G with upstream embedded point features
      - run a small platonic MLP per edge
      - max-pool over j to produce a single feature per center
    """
    def __init__(
        self,
        kv_in_channels: int,
        out_channels: int,
        embed_dim: int,
        solid_name: str,
        spatial_dims: int = 3,
        bias: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        self.spatial_dims = spatial_dims

        if kv_in_channels % self.num_G != 0:
            raise ValueError(f"kv_in_channels ({kv_in_channels}) must be divisible by group size ({self.num_G}).")
        if embed_dim % self.num_G != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by group size ({self.num_G}).")
        if out_channels % self.num_G != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by group size ({self.num_G}).")

        self.kv_per_g = kv_in_channels // self.num_G

        offset_in = self.num_G * spatial_dims
        combined_in = kv_in_channels + offset_in

        _act = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}.get(activation, nn.GELU)
        self.edge_mlp = nn.Sequential(
            PlatonicLinear(combined_in, embed_dim, solid_name, bias=bias),
            _act(),
            PlatonicLinear(embed_dim, embed_dim, solid_name, bias=bias),
        )
        self.out_proj = PlatonicLinear(embed_dim, out_channels, solid_name, bias=bias)

    def forward(
        self,
        q_pos: Tensor,
        kv_x: Tensor,
        kv_pos: Tensor,
        knn_indices: Tensor,
    ) -> Tensor:
        N_q, k = knn_indices.shape
        idx_flat = knn_indices.reshape(-1)

        kv_local = kv_x[idx_flat].view(N_q, k, -1)
        kv_local_pos = kv_pos[idx_flat].view(N_q, k, self.spatial_dims)
        rel_offset = kv_local_pos - q_pos.unsqueeze(1)

        offset_lifted = lift_vectors(rel_offset.unsqueeze(-2), self.group)

        offset_g = offset_lifted.view(N_q, k, self.num_G, self.spatial_dims)
        kv_g = kv_local.view(N_q, k, self.num_G, self.kv_per_g)
        combined_g = torch.cat([offset_g, kv_g], dim=-1)
        combined = combined_g.view(N_q, k, -1)

        edge_features = self.edge_mlp(combined)
        pooled, _ = edge_features.max(dim=1)
        return self.out_proj(pooled)
