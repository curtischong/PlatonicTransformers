import re
from collections import defaultdict

import torch


def _strip_layer_prefix(name):
    """Strip everything up to and including 'layers.N.' from a parameter name.

    Works regardless of how many prefix segments precede 'layers':
      layers.0.attention.weight          -> attention.weight
      transformer.layers.0.attention.weight -> attention.weight
      net._orig_mod.transformer.layers.3.attention.weight -> attention.weight
    """
    return re.sub(r"^.*layers\.\d+\.", "", name)


def get_mup_multipliers(base_model, main_model):
    """
    Make a dict of name:multiplier for each parameter in main model
    """
    base_shapes = _get_shapes(base_model)
    model_shapes = _get_shapes(main_model)
    basenames = set(base_shapes.keys())
    names = set(model_shapes.keys())
    assert basenames == names, (
        f"`base_shapes` has extra names {basenames - names}. " f"`shapes` has extra names {names - basenames}."
    )
    multipliers = {}
    for name, b_shape in base_shapes.items():
        multipliers[name] = _get_multiplier(b_shape, model_shapes[name])
    return multipliers


def _get_multiplier(base_dims, dims):
    # the 'multiplier' is the ratio of dim / base_dim for the **last dimension** that is infinite
    # the weight is 'matrix like' if it has >1 infinite dimension
    # eg if base_dims=[d1, d2_base] and dims=[d1, d2] we would return (d2/d2_base, False)
    num_inf_dims = 0
    multiplier = 1
    for base_dim, dim in zip(base_dims, dims):
        assert isinstance(base_dim, int), f"Unknown base_dim type: {type(base_dim)}"
        if base_dim != dim:
            num_inf_dims += 1
            multiplier = dim / base_dim
    is_matrix_like = True if num_inf_dims > 1 else False
    return (multiplier, is_matrix_like)


def mup_init(model, mup_multipliers_dict):
    for name, module in model.named_modules():
        if isinstance(module, MuReadout):
            module.width_mult = mup_multipliers_dict[f"{name}.weight"][0]
            module._rescale_parameters()
        if isinstance(module, SO3_MuReadout):
            module.width_mult = mup_multipliers_dict[f"{name}.weight"][0]
            module._rescale_parameters()
    for name, param in model.named_parameters():
        if "layers" in name:
            name = _strip_layer_prefix(name)
        if "bias" in name and name in mup_multipliers_dict:
            param.data *= mup_multipliers_dict[name][0] ** 0.5


def build_optimizer_param_groups(model, mup_multipliers_dict, decoupled_wd=False, optimizer_kwargs = None):
    """
    MuP scales the lr according to if a param is 'matrix like' or 'vector like'
    We build params_groups based on this scaled lr
    """
    if mup_multipliers_dict is None:
        # No muP: separate params into decay and no-decay groups.
        # Weight decay on: kernel/weight matrices (2D+ params in linear layers)
        # Weight decay off: biases, LayerNorm/RMSNorm weights, LayerScale gammas,
        #                   RoPE frequencies, embeddings, 1D params, FiLM proj
        #                   weights (zero-init for identity — WD pulls them back
        #                   toward 0 and works against the model learning chgspin
        #                   modulation; same rationale as LayerScale gammas).
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # No decay for: biases, norm weights, LayerScale gammas, RoPE freqs,
            # embeddings, FiLM projection weights/biases.
            if (param.ndim <= 1
                or "bias" in name
                or "norm" in name
                or "gamma" in name
                or "freqs" in name
                or "embedding" in name
                or "film_projs" in name
                or "S_triu" in name):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = []
        if decay_params:
            g = {k: v for k, v in optimizer_kwargs.items()}
            g["params"] = decay_params
            groups.append(g)
        if no_decay_params:
            g = {k: v for k, v in optimizer_kwargs.items()}
            g["params"] = no_decay_params
            g["weight_decay"] = 0.0
            groups.append(g)
        return groups

    def new_group():
        new_g = {k: v for k, v in optimizer_kwargs.items()}
        new_g["params"] = []
        return new_g

    param_groups = defaultdict(new_group)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Strip wrapper prefixes added by torch.compile, FSDP, DDP
        for prefix in ["_orig_mod.", "_fsdp_wrapped_module.", "module."]:
            if prefix in name:
                name = name.split(prefix)[-1]
        if "layers" in name:
            name = _strip_layer_prefix(name)
        multiplier, is_matrix_like = mup_multipliers_dict[name]
        if is_matrix_like:
            param_groups[multiplier]["params"].append(param)
        else:
            param_groups[1.0]["params"].append(param)

    for width_mult, group in param_groups.items():
        # Scale learning rate and weight decay accordingly
        group["lr"] /= width_mult
        if not decoupled_wd:
            group["weight_decay"] *= width_mult

    return list(param_groups.values())


def _get_shapes(model):
    """
    Returns a dictionary of name:shape for each unique layer in a model.
    If a model comprises multiple 'blocks' (eg TransformerBlocks)
    we assume every block has the same dimensions
    """
    shapes_dict = {}
    for name, param in model.named_parameters():
        if "layers.0" in name:
            name = _strip_layer_prefix(name)
            shapes_dict[name] = param.shape
        elif "layers" in name:
            name = _strip_layer_prefix(name)
            assert shapes_dict[name] == param.shape, "_get_shapes assumes all blocks have the same dimensions"
        else:
            shapes_dict[name] = param.shape
    return shapes_dict


class MuReadout(torch.nn.Linear):
    """Drop-in replacement for all output linear layers.

    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).

    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.width_mult = None
        self._has_rescaled_params = False

    def _rescale_parameters(self):
        """
        Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        assert self.width_mult is not None, "Width multiplier not set - have you called mup_init on the model?"
        if self._has_rescaled_params:
            raise RuntimeError("`_rescale_parameters` has been called once before already.")
        if self.bias is not None:
            self.bias.data *= self.width_mult**0.5
        self.weight.data *= self.width_mult**0.5
        self._has_rescaled_params = True

    def forward(self, x):
        assert self.width_mult is not None, "width multiplier not set - have you called mup_init on the model?"
        return super().forward(x / self.width_mult)


## TODO: implementing MuReadout for equivariant network ###
try:
    from nets.uma.nn.so3_layers import SO3_Linear
except ImportError:
    SO3_Linear = object
class SO3_MuReadout(SO3_Linear):
    def __init__(self, in_features: int, out_features: int, lmax: int) -> None:
        super().__init__(in_features, out_features, lmax)
        self.width_mult = None 
        self._has_rescaled_params = False
    
    def _rescale_parameters(self):
        assert self.width_mult is not None, "Width multiplier not set - have you called mup_init on the model?"
        if self._has_rescaled_params:
            raise RuntimeError("`_rescale_parameters` has been called once before already.")
        if self.bias is not None:
            self.bias.data *= self.width_mult**0.5
        self.weight.data *= self.width_mult**0.5
        self._has_rescaled_params = True
    
    def forward(self, x):
        assert self.width_mult is not None, "width multiplier not set - have you called mup_init on the model?"
        return super().forward(x / self.width_mult)

