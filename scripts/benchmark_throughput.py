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
    data_source = str(getattr(bench, "data_source", "real")).lower() if bench else "real"
    if data_source not in ("synthetic", "real"):
        raise ValueError(f"--bench.data_source must be 'synthetic' or 'real', got {data_source!r}")

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

    # Batch source.
    if data_source == "synthetic":
        single_batch = _make_synthetic_batch(n_atoms, device=device)
        batches_iter = None
        batch_label = f"synthetic, N={n_atoms} atoms in a 30 Å box (CHNO)"
    else:
        from platonic_transformers.datasets.omol import get_omol_loaders
        # Same dataloader settings as production training. dynamic_batching
        # honors training.max_atoms_per_batch from the chosen yaml.
        train_loader, _, _, _, _ = get_omol_loaders(
            root=config.dataset.data_dir,
            batch_size=config.training.batch_size,
            num_workers=int(getattr(config.system, "num_workers", 8)),
            use_charges=False,
            seed=config.seed,
            debug_subset=config.dataset.debug_subset,
            referencing=config.dataset.referencing,
            include_hof=config.dataset.include_hof,
            scale_shift=config.dataset.scale_shift,
            recalculate=config.dataset.recalculate_stats,
            use_k_hot=config.dataset.use_khot_encoding,
            dynamic_batching=bool(getattr(config.training, "dynamic_batching", False)),
            max_atoms_per_batch=getattr(config.training, "max_atoms_per_batch", None),
            max_atoms_per_batch_val=getattr(config.training, "max_atoms_per_batch_val", None),
            max_edges_per_batch=getattr(config.training, "max_edges_per_batch", None),
            max_edges_per_batch_val=getattr(config.training, "max_edges_per_batch_val", None),
            train_subdir=str(getattr(config.dataset, "train_subdir", "train_4M")),
            val_subdir=str(getattr(config.dataset, "val_subdir", "val")),
        )
        single_batch = None
        batches_iter = iter(train_loader)
        _ma = getattr(config.training, "max_atoms_per_batch", None)
        _db = getattr(config.training, "dynamic_batching", False)
        batch_label = (f"real OMol25 batches via get_omol_loaders, "
                       f"max_atoms_per_batch={_ma}, "
                       f"dynamic_batching={_db}")

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
    print(f"  data source:   {batch_label}")
    print(f"  compile:       {compiled}  (yaml training.compile)")
    print(f"  warmup steps:  {n_warmup}")
    print(f"  timed steps:   {n_timed}")
    print("=" * 60)

    def _step_fwd_bwd(b):
        e, f = model.pred_energy_and_force(b)
        loss = e.sum() + f.pow(2).sum()
        loss.backward()
        model.zero_grad(set_to_none=True)

    def _step_fwd(b):
        # Forward only (MD-inference convention from the AllScAIP paper).
        # `enable_grad` required for eSEN: its conservative forces are
        # computed via autograd.grad inside the MLP_EFS_Head, so the grad
        # tape must be live even when we never call .backward(). For direct-
        # force models (Platonic / AllScAIP variant) it's harmless.
        with torch.enable_grad():
            e, f = model.pred_energy_and_force(b)
        # Drop autograd refs so activations from this step are freed before
        # the next forward (otherwise eSEN's grad-graph piles up).
        del e, f

    def _atoms_in(b) -> int:
        # Works for both Batch and fairchem AtomicData: pos has one row per atom.
        return int(b.pos.shape[0])

    def _time(step_fn, label):
        """Warmup then time. Returns (total_wall_seconds, atoms_processed)."""
        if data_source == "synthetic":
            # Same batch reused for warmup + timed — most reps go in compile/
            # autotune the first time, steady-state by step ~5.
            for _ in range(n_warmup):
                step_fn(single_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_timed):
                step_fn(single_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return time.perf_counter() - t0, n_timed * _atoms_in(single_batch)

        # data_source == "real"
        # Warmup. Each batch from the dataloader has a different shape, so
        # compile may recompile a few times before stabilising.
        for _ in range(n_warmup):
            b = next(batches_iter).to(device)
            step_fn(b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        # Timed.
        atoms_seen = 0
        t0 = time.perf_counter()
        for _ in range(n_timed):
            b = next(batches_iter).to(device)
            step_fn(b)
            atoms_seen += _atoms_in(b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, atoms_seen

    results = []
    for step_fn, lbl in [(_step_fwd_bwd, "forward + backward (training cost)"),
                        (_step_fwd, "forward only (MD-inference cost)")]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            total, atoms = _time(step_fn, lbl)
        except RuntimeError as exc:
            # eSEN's conservative forces use autograd.grad inside the head;
            # `loss.backward()` on the result is a double-backward that
            # torch.compile+aot_autograd does NOT support. Catch and skip the
            # offending mode so the other one (and the next model) still runs.
            msg = str(exc)
            print()
            print(f"--- {lbl} ---")
            print(f"  SKIPPED: {type(exc).__name__}: {msg.splitlines()[-1] if msg else ''}")
            if "double backward" in msg:
                print(f"  (this is the known torch.compile + autograd.grad-inside-forward")
                print(f"   incompatibility; rerun this mode with --training.compile=false)")
            results.append({"lbl": lbl, "skipped": True})
            continue
        peak = (torch.cuda.max_memory_allocated() / 1024**3
                if device.type == "cuda" else 0.0)
        ms_per_step = (total / n_timed) * 1000.0
        steps_per_sec = n_timed / total
        atoms_per_sec = atoms / total
        avg_atoms_per_step = atoms / n_timed
        # ns/day at 1 fs MD timestep — meaningful when the batch is one
        # fixed-N system (synthetic), ambiguous when batches vary (real).
        ns_per_day_at_1fs = 1e-6 * steps_per_sec * 86400.0
        results.append({
            "lbl": lbl, "skipped": False,
            "total": total, "ms_per_step": ms_per_step,
            "steps_per_sec": steps_per_sec, "atoms_per_sec": atoms_per_sec,
            "avg_atoms": avg_atoms_per_step,
            "ns_per_day": ns_per_day_at_1fs, "peak": peak,
        })

    for r in results:
        print()
        print(f"--- {r['lbl']} ---")
        if r.get("skipped"):
            continue
        print(f"  total wall:    {r['total']:.3f} s")
        print(f"  ms/step:       {r['ms_per_step']:.2f}")
        print(f"  steps/sec:     {r['steps_per_sec']:.2f}")
        print(f"  avg atoms/step:{r['avg_atoms']:,.0f}")
        print(f"  atoms/sec:     {r['atoms_per_sec']:,.0f}")
        if data_source == "synthetic":
            print(f"  ns/day @1fs:   {r['ns_per_day']:.3f}  "
                  f"(fixed-N synthetic; meaningful for MD inference)")
        if device.type == "cuda":
            print(f"  peak VRAM:     {r['peak']:.2f} GiB")
    print()


if __name__ == "__main__":
    main()
