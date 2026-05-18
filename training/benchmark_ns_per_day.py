"""Inference-throughput benchmark in ns/day (MD convention).

Methodology matches Qu et al. 2026 (AllScAIP, arXiv:2603.06567):
single GPU, one batched system of N atoms, forward only, graph generation off
(for our models that means otf_graph is whatever the config says — we don't
exclude its cost). At a 1-fs MD timestep, one forward = one integrator step,
so ns/day = 86400 * dt[s] / (s/step) = 8.64e-2 / (s/step) at dt=1 fs.

Both architectures supported:
  * platoformer  — direct forces, no autograd-grad inside forward.
  * esen / uma   — direct_forces=False, forces produced via torch.autograd.grad
                   inside MLP_EFS_Head. Requires grad-enabled context.

We never enter torch.no_grad / inference_mode (which would break eSEN). For
the direct-force models the overhead is small. Hydra config tree is the same
as train_omol.py so precision/matmul knobs propagate.

Usage (snellius example):
    python benchmark_ns_per_day.py \\
        +precision=fp32_baseline force_field_module=platoformer data=omol_4m \\
        force_field_module.net.hidden_dim=1920 \\
        force_field_module.net.nhead=60 \\
        force_field_module.net.num_layers=8 \\
        force_field_module.net.ffn_dim_factor=2 \\
        force_field_module.net.layer_scale_init_value=1e-4 \\
        force_field_module.net.activation=sin \\
        +force_field_module.net.readout_activation=sin \\
        force_field_module.net.rope_sigma=2.0 \\
        +force_field_module.net.rope_on_values=true \\
        force_field_module.net.attention_backend=flash \\
        force_field_module.net.chgspin_mode=add \\
        data.datamodule.max_atoms_per_batch=1000 \\
        data.datamodule.max_atoms_per_batch_val=1000 \\
        data.datamodule.dynamic_batching=true \\
        +bench.n_warmup=10 +bench.n_timed=50 +bench.dt_fs=1.0 \\
        wandb.use_wandb=False exp_name=bench
"""
import logging
import time

import hydra
import rootutils
import torch
from omegaconf import OmegaConf
from fairchem.core.datasets.atomic_data import AtomicData

# Same torch.load monkeypatch as train_omol.py for cross-version checkpoint safety.
_orig_torch_load = torch.load
def _patched_torch_load(*a, **k):
    k["weights_only"] = False
    return _orig_torch_load(*a, **k)
torch.load = _patched_torch_load

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging.getLogger(__name__)


def _atoms_in_batch(batch) -> int:
    return int(batch.pos.shape[0])


def _make_single_system(N: int, device, box_size: float = 30.0, seed: int = 0):
    """Construct one connected system of N atoms — matches the paper's protocol
    (one molecule of N atoms, not a batch of small ones). Atom identities sampled
    from a biomolecular C/H/N/O mix; positions uniform in a box. Random weights
    are fine for throughput measurement — the model doesn't care about chemistry.

    Returns a fairchem AtomicData populated with empty edge_index/cell_offsets/nedges
    so otf_graph=True (in the model config) will build the neighbor list at first
    forward. PBC fields are zeroed; use_pbc=False in the model config skips them.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = (torch.rand(N, 3, generator=g) * box_size).to(device)
    z_pool = torch.tensor([6, 1, 7, 8], dtype=torch.long)  # C, H, N, O
    z_probs = torch.tensor([0.50, 0.35, 0.08, 0.07])
    idx = torch.multinomial(z_probs, N, replacement=True, generator=g)
    atomic_numbers = z_pool[idx].to(device)

    data = AtomicData(
        pos=pos,
        atomic_numbers=atomic_numbers,
        cell=torch.zeros(1, 3, 3, dtype=torch.float32, device=device),
        pbc=torch.zeros(1, 3, dtype=torch.bool, device=device),
        natoms=torch.tensor([N], dtype=torch.long, device=device),
        edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
        # cell_offsets dtype must match pos (fairchem validate() asserts).
        cell_offsets=torch.zeros(0, 3, dtype=torch.float32, device=device),
        nedges=torch.tensor([0], dtype=torch.long, device=device),
        charge=torch.tensor([0.0], dtype=torch.float32, device=device),
        spin=torch.tensor([1.0], dtype=torch.float32, device=device),
        fixed=torch.zeros(N, dtype=torch.long, device=device),
        tags=torch.zeros(N, dtype=torch.long, device=device),
        batch=torch.zeros(N, dtype=torch.long, device=device),
    )
    # Some PlatonicForceField paths expect data.z; mirror atomic_numbers.
    data.z = atomic_numbers
    return data


@hydra.main(config_path="./configs/", config_name="train_omol", version_base="1.2")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(cfg.cudnn_deterministic)

    bench_cfg = cfg.get("bench", None)
    n_warmup = int(bench_cfg.get("n_warmup", 10) if bench_cfg else 10)
    n_timed = int(bench_cfg.get("n_timed", 50) if bench_cfg else 50)
    dt_fs = float(bench_cfg.get("dt_fs", 1.0) if bench_cfg else 1.0)
    mode = str(bench_cfg.get("mode", "batched") if bench_cfg else "batched")
    n_atoms_single = int(bench_cfg.get("n_atoms", 1000) if bench_cfg else 1000)

    log.info("Instantiating model <%s>", cfg.force_field_module._target_)
    model = hydra.utils.instantiate(cfg.force_field_module)
    net = model.net.to(device).eval()
    n_params = sum(p.numel() for p in net.parameters())

    # Optional torch.compile for forward-only inference benchmark. The training-time
    # compile hook in omol_module only fires on stage=="fit"; for the benchmark we
    # apply it directly here so MODE=single/batched runs go through the compiled path.
    compile_flag = bool(bench_cfg.get("compile", False) if bench_cfg else False)
    if compile_flag:
        compile_mode = str(bench_cfg.get("compile_mode", "default"))
        compile_dynamic = bench_cfg.get("compile_dynamic", None)
        log.info("torch.compile(net, mode=%s, dynamic=%s)", compile_mode, compile_dynamic)
        net = torch.compile(net, mode=compile_mode, dynamic=compile_dynamic)

    if mode == "single":
        log.info("Building synthetic single-system: N=%d atoms", n_atoms_single)
        batch = _make_single_system(n_atoms_single, device=device)
    else:
        log.info("Instantiating datamodule <%s>", cfg.data.datamodule._target_)
        dm = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader)).to(device)
        batch.z = batch.atomic_numbers.long()
    n_atoms = _atoms_in_batch(batch)
    n_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1

    print(f"\n=== System ===")
    print(f"model:           {cfg.force_field_module.net._target_}")
    print(f"params:          {n_params:,}")
    print(f"device:          {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})")
    print(f"precision:       trainer.precision={cfg.trainer.precision}  matmul={cfg.matmul_precision}")
    print(f"torch.compile:   {compile_flag}")
    print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"mode:            {mode}  (single = one molecule of N atoms; batched = dataloader)")
    print(f"atoms in batch:  {n_atoms}")
    print(f"graphs in batch: {n_graphs}")
    print(f"warmup steps:    {n_warmup}")
    print(f"timed steps:     {n_timed}")
    print(f"MD timestep:     {dt_fs:.2f} fs")

    # Warmup.
    for _ in range(n_warmup):
        out = net(batch)
        # Touch forces so any lazy autograd.grad inside the head executes.
        _ = out["forces"]
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed forward (no backward; MD inference convention).
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        out = net(batch)
        # Keep the autograd.grad path materialized in eSEN.
        _ = out["forces"]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_sec = t1 - t0
    sec_per_step = total_sec / n_timed
    steps_per_sec = 1.0 / sec_per_step
    ns_per_day = dt_fs * 1e-6 * steps_per_sec * 86400  # dt[fs]→ns
    atom_ns_per_day = n_atoms * ns_per_day

    print(f"\n=== Throughput ===")
    print(f"total wall (s):      {total_sec:.3f}")
    print(f"ms/step:             {sec_per_step*1000:.2f}")
    print(f"steps/sec:           {steps_per_sec:.2f}")
    print(f"ns/day @ dt={dt_fs:.2f} fs: {ns_per_day:.3f}")
    print(f"atom-ns/day:         {atom_ns_per_day:.3e}")

    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak VRAM (GiB):     {peak_gib:.2f}")


if __name__ == "__main__":
    main()
