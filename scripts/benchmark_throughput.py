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

# Same precision + cudnn knobs as the training entrypoint.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
    print(f"  warmup steps:  {n_warmup}")
    print(f"  timed steps:   {n_timed}")
    print("=" * 60)

    def _step():
        e, f = model.pred_energy_and_force(batch)
        # Cheap pseudo-loss; we only care about touching the gradient path.
        loss = e.sum() + f.pow(2).sum()
        loss.backward()
        model.zero_grad(set_to_none=True)

    # Warmup (includes torch.compile pass, dynamo recompiles, kernel autotune).
    for _ in range(n_warmup):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    ms_per_step = (total / n_timed) * 1000.0
    steps_per_sec = n_timed / total
    atoms_per_sec = n_atoms * steps_per_sec
    # 1 fs MD timestep convention (matches AllScAIP paper for inference;
    # this script is fwd+bwd so ns/day here = "training-steps as fake-MD steps").
    ns_per_day_at_1fs = 1e-6 * steps_per_sec * 86400.0

    print()
    print("--- throughput (forward + backward) ---")
    print(f"  total wall:    {total:.3f} s")
    print(f"  ms/step:       {ms_per_step:.2f}")
    print(f"  steps/sec:     {steps_per_sec:.2f}")
    print(f"  atoms/sec:     {atoms_per_sec:,.0f}")
    print(f"  ns/day @1fs:   {ns_per_day_at_1fs:.3f}  (fwd+bwd, not pure inference)")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  peak VRAM:     {peak:.2f} GiB")
    print()


if __name__ == "__main__":
    main()
