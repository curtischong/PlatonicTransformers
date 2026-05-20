"""Throughput benchmark for Platonic Transformer / AllScAIP / eSEN on OMol25.

Runs forward + backward on a single synthetic N-atom system, times the
steady-state cost, and reports ms/step, atoms/sec and ns/day. Lets you
compare model variants on the same hardware without a full training run.

Usage (from repo root, inside the venv):

    # Platonic Transformer (qcczbpfn recipe — attention, no local_global)
    python scripts/benchmark_throughput.py --config configs/omol.yaml

    # AllScAIP variant of Platonic Transformer (dual local→global blocks)
    python scripts/benchmark_throughput.py --config configs/omol.yaml \\
        --model.dense_mode=false --model.local_global=true \\
        --model.interaction_radius=2.0

    # eSEN baseline (conservative forces via autograd.grad)
    python scripts/benchmark_throughput.py --config configs/omol_esen.yaml

CLI overrides supported via the same parser as `mains/main_omol.py`. The
synthetic batch is one molecule of N atoms (default N=1000) with CHNO
composition; charge/spin are zeroed. Random positions/elements are fine
for throughput — the model doesn't care about chemistry.
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from platonic_transformers.utils.config_loader import (
    get_arg_parser, load_with_defaults,
)
from platonic_transformers.datasets.omol import Batch

# Same precision + cudnn + dynamo knobs as the training entrypoint
# (mains/main_omol.py module-level). Needed for torch.compile to behave on
# dynamic shapes without dropping to eager mode.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import torch._dynamo as _dynamo  # noqa: E402
_dynamo.config.cache_size_limit = 256
_dynamo.config.force_parameter_static_shapes = False
_dynamo.config.capture_scalar_outputs = True


def _make_synthetic_batch(n_atoms: int, device, dtype=torch.float32,
                          box: float = 30.0, seed: int = 0):
    """One molecule of `n_atoms` atoms with CHNO composition + zero charge/spin.

    Returns the same `Batch` class our OMolDataset emits, padded with the
    fairchem-style fields (`natoms`, `pbc`, `cell`) so the eSEN backbone
    also accepts it.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = (torch.rand(n_atoms, 3, generator=g) * box).to(device).to(dtype)
    z_pool = torch.tensor([6, 1, 7, 8], dtype=torch.long)        # C H N O
    z_probs = torch.tensor([0.50, 0.35, 0.08, 0.07])
    idx = torch.multinomial(z_probs, n_atoms, replacement=True, generator=g)
    atomic_numbers = z_pool[idx].to(device)

    data_dict = {
        "pos": pos,
        "x": torch.zeros(n_atoms, 92, device=device, dtype=dtype),
        "energy": torch.zeros(1, device=device, dtype=dtype),
        "forces": torch.zeros(n_atoms, 3, device=device, dtype=dtype),
        "batch": torch.zeros(n_atoms, dtype=torch.long, device=device),
        "edge_index": None,
        "edge_attr": None,
        "name": ["synth"],
        "smiles": [""],
        "composition": [""],
        "idx": [0],
        "num_atoms": torch.tensor([n_atoms], dtype=torch.long, device=device),
        "charges": torch.zeros(n_atoms, dtype=dtype, device=device),
        "atomic_numbers": atomic_numbers,
        "charge": torch.zeros(1, dtype=torch.long, device=device),
        "spin": torch.zeros(1, dtype=torch.long, device=device),
        "cum_nodes": torch.tensor(n_atoms),
    }
    batch = Batch(data_dict)
    # fairchem-AtomicData-shaped extras for eSCNMDBackbone (otf_graph builds edges).
    batch.natoms = data_dict["num_atoms"]
    batch.cell = torch.zeros(1, 3, 3, device=device, dtype=dtype)
    batch.pbc = torch.zeros(1, 3, dtype=torch.bool, device=device)
    return batch


def main():
    parser = get_arg_parser(default_config_path="configs/omol.yaml")
    args, unknown_args = parser.parse_known_args()
    config = load_with_defaults(
        dataset_config=args.config, cli_args=unknown_args,
    )

    bench = getattr(config, "bench", None)
    n_warmup = int(getattr(bench, "n_warmup", 10)) if bench else 10
    n_timed = int(getattr(bench, "n_timed", 50)) if bench else 50
    n_atoms = int(getattr(bench, "n_atoms", 1000)) if bench else 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the right Lightning module without ever invoking Lightning.
    model_name = str(getattr(config.model, "name", "platoformer")).lower()
    if model_name == "esen":
        from mains.main_omol import ESENModel
        model = ESENModel(config).to(device)
    else:
        from mains.main_omol import OMolModel
        model = OMolModel(config).to(device)
    model.train()  # train mode → autograd graph built (needed for eSEN forces)

    # Compile self.net the same way OMolModel.setup() does on stage="fit".
    # We bypass Lightning here, so we have to run this step manually; without
    # it the benchmark times eager-mode forwards, which under-reports throughput
    # on the same recipe that qcczbpfn runs with `compile=true`. Honors yaml's
    # `training.compile` flag and CLI overrides.
    if bool(getattr(config.training, "compile", False)):
        cmode = str(getattr(config.training, "compile_mode", "default"))
        cdyn = getattr(config.training, "compile_dynamic", None)
        try:
            model.net = torch.compile(model.net, mode=cmode, dynamic=cdyn)
            compiled = True
        except Exception as exc:
            print(f"[benchmark] torch.compile failed ({exc!r}); running eager.")
            compiled = False
    else:
        compiled = False

    n_params = sum(p.numel() for p in model.parameters())

    batch = _make_synthetic_batch(n_atoms, device=device)

    print()
    print("=" * 60)
    print(f"  model:         {model_name}")
    print(f"  config:        {args.config}")
    if model_name == "platoformer":
        m = config.model
        print(f"  hidden_dim:    {getattr(m, 'hidden_dim', '?')}")
        print(f"  num_layers:    {getattr(m, 'num_layers', '?')}")
        print(f"  attention:     {getattr(m, 'attention', '?')}")
        print(f"  attn_backend:  {getattr(m, 'attention_backend', '?')}")
        print(f"  dense_mode:    {getattr(m, 'dense_mode', '?')}")
        print(f"  local_global:  {getattr(m, 'local_global', False)}")
        print(f"  interaction_radius: {getattr(m, 'interaction_radius', None)}")
    print(f"  params:        {n_params:,}")
    print(f"  device:        {device}  "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"  n_atoms:       {n_atoms}")
    print(f"  compile:       {compiled}  (yaml training.compile)")
    print(f"  warmup steps:  {n_warmup}")
    print(f"  timed steps:   {n_timed}")
    print("=" * 60)

    def _step_fwd_bwd():
        e, f = model.pred_energy_and_force(batch)
        # Cheap pseudo-loss; we only care about touching the gradient path.
        loss = e.sum() + f.pow(2).sum()
        loss.backward()
        model.zero_grad(set_to_none=True)

    def _step_fwd():
        # Forward only (MD-inference convention from the AllScAIP paper).
        # `enable_grad` is required for eSEN: its conservative forces are
        # computed via autograd.grad inside the MLP_EFS_Head, so the grad
        # tape must be live even when we never call .backward(). For direct-
        # force models (Platonic / AllScAIP variant) it's harmless.
        with torch.enable_grad():
            e, f = model.pred_energy_and_force(batch)
        # Drop autograd refs so activations from this step are freed before
        # the next forward (otherwise eSEN's grad-graph piles up).
        del e, f

    def _time(step_fn, label):
        # Warmup (compile + dynamo recompiles + kernel autotune).
        for _ in range(n_warmup):
            step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        # Timed.
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_timed):
            step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t1 - t0, label

    results = []
    for step_fn, lbl in [(_step_fwd_bwd, "forward + backward (training cost)"),
                        (_step_fwd, "forward only (MD-inference cost)")]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total, _ = _time(step_fn, lbl)
        peak = (torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda" else 0.0)
        ms_per_step = (total / n_timed) * 1000.0
        steps_per_sec = n_timed / total
        atoms_per_sec = n_atoms * steps_per_sec
        ns_per_day_at_1fs = 1e-6 * steps_per_sec * 86400.0
        results.append((lbl, total, ms_per_step, steps_per_sec,
                        atoms_per_sec, ns_per_day_at_1fs, peak))

    for lbl, total, ms, sps, aps, nspd, peak in results:
        print()
        print(f"--- {lbl} ---")
        print(f"  total wall:    {total:.3f} s")
        print(f"  ms/step:       {ms:.2f}")
        print(f"  steps/sec:     {sps:.2f}")
        print(f"  atoms/sec:     {aps:,.0f}")
        print(f"  ns/day @1fs:   {nspd:.3f}")
        if device.type == "cuda":
            print(f"  peak VRAM:     {peak:.2f} GiB")
    print()


if __name__ == "__main__":
    main()
