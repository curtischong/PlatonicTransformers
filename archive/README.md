# Platonic Transformer for OMol25

Training, evaluation, and leaderboard submission for the **Platonic Transformer** on the [OMol25](https://fair-chem.github.io/molecules/datasets/omol25.html) molecular force field benchmark.

**Goal:** Achieve competitive results on the [FAIR Chemistry Leaderboard](https://huggingface.co/spaces/facebook/fairchem_leaderboard) using equivariant Platonic Transformers.

## Quick start (existing environment)

```bash
cd training
export DATA_PATH=/path/to/omol25       # parent dir containing open_mol/{train_4M,val}
python train_omol.py \
    force_field_module=platoformer \
    data=omol_4m \
    force_field_module.compile=true \
    force_field_module.net.hidden_dim=1728 \
    force_field_module.net.nhead=36 \
    force_field_module.net.num_layers=12 \
    force_field_module.net.solid_name=tetrahedron \
    force_field_module.net.attention_backend=flash \
    force_field_module.net.chgspin_mode=add \
    force_field_module.train_augmentation=o3 \
    +force_field_module.optimizer.r=2.0 \
    force_field_module.train_rmsd=1.433569 \
    exp_name=platoformer-omol \
    wandb.group=platonic-omol
```

For long-running cluster jobs, use the SLURM scripts in `scripts/`.

## Setting up a fresh cluster

The setup has four stages: clone, venv, flash-attn build, data. Stages 3 and 4 each take 30-60 minutes and can run in parallel.

### 1. Clone the repo

```bash
git clone git@github.com:ebekkers/platonic-omol.git
cd platonic-omol
git checkout upstream-port-pt2     # or main, once merged
```

### 2. Create a venv and install Python deps

We use Python 3.12 + uv (or plain pip — uv is just faster).

```bash
uv venv --python 3.12 venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
```

**Important: the `lightning` PyPI package is currently quarantined**, so a plain `pip install lightning` fails with `No matching distribution found`. Build it from source instead:

```bash
git clone --branch 2.5.5 --depth 1 https://github.com/Lightning-AI/pytorch-lightning.git /tmp/pytorch-lightning
cd /tmp/pytorch-lightning
PACKAGE_NAME=lightning pip install --no-deps .
cd -
```

This produces both the `lightning` and `lightning_fabric` namespaces (what our code imports). Then install the rest:

```bash
pip install hydra-core omegaconf rootutils humanize ase lmdb schedulefree wandb e3nn matplotlib Pillow
pip install 'git+https://github.com/facebookresearch/fairchem.git@fairchem_core-2.0.0#subdirectory=packages/fairchem-core'
```

`torch-cluster` and `torch-scatter` are listed in `requirements.txt` but **not actually needed** for PT-2: the only `torch_cluster.knn_graph` usage is wrapped in a try/except and `torch_scatter` calls have been replaced with native `Tensor.scatter_add_`. Skip them — they need GPU-side compilation that often fails on login nodes.

Verify the import chain:

```bash
python -c "import torch, lightning, hydra, schedulefree, fairchem; \
  from fairchem.core.datasets import AseDBDataset; \
  print('torch', torch.__version__, 'cuda', torch.version.cuda); \
  print('lightning', lightning.__version__); \
  print('AseDBDataset OK')"
```

### 3. Install flash-attn

**Try a pre-built wheel first** (fast path, works whenever your torch+CUDA combo matches an upstream release). Look up your torch ABI:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
  'cxx11abi', torch._C._GLIBCXX_USE_CXX11_ABI)"
```

Then pick a matching wheel from [Dao-AILab/flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases). Example for torch 2.6 + cu12 + Python 3.12 + cxx11abiFALSE (what fairchem-core 2.0.0 pulls in):

```bash
pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'
```

Verify:

```bash
python -c "from flash_attn import flash_attn_varlen_func; import flash_attn; print('flash_attn', flash_attn.__version__, 'OK')"
```

**Fallback: build from source** (if no matching wheel exists, e.g., torch 2.11+ on Snellius). **Use a GPU compute node, not the login node** — CUDA dev tools and ~120GB RAM for the compile. Set `TORCH_CUDA_ARCH_LIST` to your GPU's arch:

| GPU | arch | TORCH_CUDA_ARCH_LIST |
|---|---|---|
| H100 (Snellius gpu_h100) | sm_90 | "9.0" |
| RTX 6000 Ada (hipster performance) | sm_89 | "8.9" |
| L4 (hipster capacity) | sm_89 | "8.9" |
| A100 | sm_80 | "8.0" |

Adapt `scripts/install_flash_attn.sh` (templated for Snellius); key env vars:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"                # 9.0 for H100
export MAX_JOBS=2                                # 4 OOM'd at 32G; 2 fits in ~120G
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export CUDA_HOME=/path/to/cuda                   # e.g. $EBROOTCUDA on Snellius
pip install flash-attn --no-build-isolation
```

Source builds take 30-90 minutes.

### 4. Download OMol25-4M data (~45GB)

OMol25-4M is the curated 4M-sample subset eSEN uses for fast iteration. Both files are on FAIR's public file server (no HF auth needed):

```bash
mkdir -p $DATA_DIR/open_mol     # e.g. DATA_DIR=/scratch/$USER/omol
cd $DATA_DIR/open_mol
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz
tar -xzf train_4M.tar.gz && rm train_4M.tar.gz
tar -xzf val.tar.gz && rm val.tar.gz
```

After extraction you'll have `train_4M/data*.aselmdb` (~22GB, 4M samples) and `val/data*.aselmdb` (~20GB, 2.76M heldout samples). Set:

```bash
export DATA_PATH=$DATA_DIR     # parent of open_mol/
```

**Don't confuse this with `OMol25 neutral_train`** — that's a different curation (charge=0 only). The Hydra `data=omol_4m` config expects the 4M shards above; `data=omol` expects the neutral split.

### 5. Smoke test

A 1-epoch dry-run on a tiny model verifies the install end-to-end:

```bash
cd training
python train_omol.py \
    force_field_module=platoformer \
    data=omol_4m \
    data.datamodule.batch_size.train=4 \
    force_field_module.compile=false \
    force_field_module.net.attention_backend=flash \
    force_field_module.net.chgspin_mode=add \
    force_field_module.net.num_layers=2 \
    force_field_module.net.hidden_dim=144 \
    force_field_module.net.nhead=4 \
    +trainer.limit_train_batches=2 \
    +trainer.limit_val_batches=2 \
    trainer.max_epochs=1 \
    wandb.use_wandb=false \
    exp_name=smoke
```

Should finish in <2 minutes on any modern GPU.

## Recent fixes

- **2026-05-11 (`deeaea2`)** — fixed a dtype mismatch in
  `nets/platoformer/platoformer.py` where the readout subnetwork hard-cast inputs
  to `.double()` but left the readout weights in whatever dtype the model was
  instantiated with. Default fp32 launches crashed with `expected mat1 and mat2
  to have the same dtype, but got: double != float`. The fix matches readout
  weight dtype, so fp32 stays fp32 and `+precision=fp64_baseline` still upcasts
  cleanly.
- **2026-05-11 (this commit)** — added direct-vs-conservative branching in
  `nets/uma/model.py`. Conservative (autograd-grad) forces use `MLP_EFS_Head`;
  direct forces use a separate `Linear_Energy_Head` + `Linear_Force_Head` pair.
  The earlier wrapper assumed `MLP_EFS_Head` always emits a `forces` key, which
  isn't true in direct mode — fixed by reading from the appropriate head.

If you pulled before May 11, just `git pull origin main`.

## eSEN baseline (fairchem)

Two paths to compare against eSEN:

**(a) Use the released checkpoint** — fastest way to get the paper's published
eSEN-sm or eSEN-md. The OMol25 weights live on Hugging Face behind a gated repo
(accept terms at https://huggingface.co/facebook/OMol25):

```bash
export HF_TOKEN=<your-hf-token>   # https://huggingface.co/settings/tokens
python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('facebook/OMol25', 'checkpoints/esen_sm_direct_all.pt'))"
```

The downloaded checkpoint is a `fairchem.core.units.mlip_unit.api.inference.MLIPInferenceCheckpoint`;
load with `torch.load(path, map_location='cpu', weights_only=False)` and read
`ckpt.model_config["backbone"]` for the full hyperparameter dict. The released
eSEN-sm-direct is **6,333,093 params**: `sphere_channels=128, hidden_channels=128,
lmax=2, mmax=2, num_layers=5, direct_forces=true`. eSEN-md-direct is 50.67M: same
width/mmax, but `lmax=4`, `num_layers=10`. eSEN-sm-conserving has effectively the
same architecture (just trained with conservative loss).

**(b) Train from scratch** — for an apples-to-apples comparison against PT under
matched data, optimizer, and precision. Two configs are provided:

- `force_field_module=esen` — Ngo & Ravanbakhsh's smaller default (sphere=32,
  lmax=4, 12 layers, 3.4M params). Slower per atom than the paper's eSEN-sm
  because higher lmax dominates eSCN compute.
- `force_field_module=esen_sm` — **paper-matching** config (sphere=128, lmax=2,
  5 layers, `direct_forces=true`, ~6.07M params). Mirrors the released
  `esen_sm_direct_all.pt` checkpoint on the 3 dominant architectural knobs.
  `torch.compile=true` by default.

**Direct vs conservative — pick direct for training.** Conservative
(`direct_forces=false`) computes forces by `torch.autograd.grad(energy, pos)`.
That makes training a **double-backward** (forces are themselves a backward;
loss requires another backward through them), which:

- triples activation memory vs a direct-force model (eSEN-sm conservative OOMs
  at `max_atoms=8000` on a 96GB H100; direct fits 20000+);
- breaks `torch.compile` (donated-buffer optimization assumes single backward).

The published AllScAIP Figure 5 "filled-circle direct" eSEN points are direct
variants — that's what we're comparing against, so `direct_forces=true` is both
the cheaper and the more apples-to-apples choice.

Launch the 20-epoch eSEN-sm training-from-scratch run on Snellius:

```bash
sbatch scripts/run_esen_sm_20ep.sh                        # default max_atoms=20000
sbatch --export=ALL,MAX_ATOMS=12000 scripts/run_esen_sm_20ep.sh
```

`MAX_ATOMS` is parametric (default 20000 — the largest batch we've validated
on H100 with direct + compile). Recipe mirrors PT's: AdamW +
cosine_annealing_ws, lr=5e-4, wd=1e-5, 1% fractional warmup, dynamic batching,
fp32 precision.

## Apples-to-apples PT-vs-eSEN-sm comparison (20 epoch, max_atoms=20000)

Submit both runs side-by-side on the same node config (1× H100, 5d cap):

```bash
# eSEN-sm direct + compile, 20ep, max_atoms=20000 (~14h projected wall)
sbatch scripts/run_esen_sm_20ep.sh

# PT-2 sin/sin rs=2.0 EMA=0.99, 20ep, max_atoms=20000 (matches eSEN batch size)
sbatch --export=ALL,MAX_ATOMS=20000,WD=1e-5,LAYER_SCALE=1e-4,FFN_FACTOR=2,\
ACTIVATION=sin,READOUT_ACTIVATION=sin,ROPE_SIGMA=2.0,EMA_DECAY=0.99 \
  --job-name=PT2-rs2-ema0.99-20ep-n20000 \
  scripts/run_pt2_h1920_l8_20ep_fp32.sh
```

Both share `omol-leaderboard/scaling-laws-symmetry` W&B project under the
`pt2-vs-esen-sm-direct-n20000` group.

## Inference throughput benchmark

`benchmark_ns_per_day.py` measures `ns/day` on a single H100 at N=1000 atoms,
forward-only, fp32 + TF32. Methodology mirrors Qu et al. 2026 (AllScAIP) Table 2.
Optional `torch.compile`, `cudnn.benchmark`, and `activation_checkpointing` knobs
let you reproduce paper-style "fast" timings:

```bash
# Eager (legacy) — matches our pre-2026-05-11 numbers
sbatch --export=ALL,MODEL=platoformer scripts/run_benchmark_ns_per_day.sh
sbatch --export=ALL,MODEL=esen        scripts/run_benchmark_ns_per_day.sh

# Paper-matching eSEN-sm architecture (sphere=128, lmax=2, L=5, ~6M params)
sbatch --export=ALL,MODEL=esen_paper scripts/run_benchmark_ns_per_day.sh

# Compiled + cudnn.benchmark + AC off (paper-style fast)
sbatch --export=ALL,MODEL=esen_paper,COMPILE=true,AC=false,CUDNN_BENCH=true \
  scripts/run_benchmark_ns_per_day.sh
sbatch --export=ALL,MODEL=platoformer,COMPILE=true,CUDNN_BENCH=true \
  scripts/run_benchmark_ns_per_day.sh
```

`torch.compile` adds ~5-10 min of first-call inductor codegen but cuts ms/step
roughly 1.4-1.7× on both eSEN and PT recipes.

**Reference results** (single H100, N=1000 atoms, single-system mode, fp32 + TF32,
forward-only, dt=1 fs):

| Recipe | Params | ms/step | ns/day | atom-ns/day | Notes |
|---|---|---|---|---|---|
| PT eager (`MODEL=platoformer`) | 18.2M | 24.7 | 3.49 | 3.5e3 | flash attention |
| **PT compiled** | 18.2M | **17.31** | **4.99** | 5.0e3 | + cudnn.benchmark; 1.43× over eager |
| eSEN Ngo default (`MODEL=esen`, sphere=32, lmax=4, L=12, conservative) | 3.4M | 140.0 | 0.62 | 6.2e2 | tiny baseline; misconfigured arch |
| eSEN paper-sized conservative (`esen_paper` w/ `direct_forces=false`) eager | 6.07M | 70.9 | 1.22 | 1.2e3 | matches HF eSEN-sm width/depth |
| eSEN paper-sized conservative compiled (AC=off, cudnn.bench) | 6.07M | 42.35 | 2.04 | 2.0e3 | 1.67× over eager; autograd.grad path partially compiled |
| **eSEN paper-sized direct + compile** (AC=off, cudnn.bench) | 6.04M | **17.07** | **5.06** | 5.1e3 | direct forces remove the autograd-grad in inference too |

For context, the published AllScAIP numbers (H200, also single-system) put:
- AllScAIP-sm 35M at 2.279 ns/day, AllScAIP-md 85M at 1.124 ns/day (Table 2)
- eSEN-sm 6M direct at ~3 ns/day (Figure 5, visual read; H200 → ~+30% over H100 for eSCN-style)

After H100→H200 adjustment (~no boost for attention models, ~+30% for eSCN/eSEN),
**both our compiled PT and our compiled eSEN-sm-direct land near 5 ns/day H100**,
≈ 5.0 / ≈ 6.6 ns/day H200-equivalent. PT is **~2× the published AllScAIP-sm**
at half the parameter count; eSEN-sm-direct is **~2× the published eSEN-sm**
at the same parameter count (within the small 6.04M vs 6.33M mismatch and
measurement noise). The published number is plausibly graph-gen-included while
ours subtracts it via `compile=true` warmup.

Notes:
- PT's `torch.compile` reports a graph break inside the flash-attn dispatch
  (`batch.max().item()` in `conv.py`). Speedup is still ~1.4×; if you want to
  squeeze more, try launching with `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`.
- eSEN-direct's first compile run is slow (~10-15 min inductor codegen on the
  first forward) because there is no `autograd.grad` graph break, so the whole
  backbone + dual heads compile as one graph. Subsequent calls hit the inductor
  cache and start in seconds.

## Repository structure

```
platonic-omol/
├── training/                   # Training code
│   ├── train_omol.py           # Hydra entry point
│   ├── nets/platoformer/       # PlatoFormer model (ported from niazoys/PlatonicTransformers)
│   ├── src/                    # Lightning datamodule + module
│   ├── configs/                # Hydra configs
│   └── utils/                  # Logging utilities
├── scripts/                    # SLURM launch scripts (one per recipe)
├── eval/                       # Leaderboard submission tools
├── requirements.txt
└── README.md
```

## Model

The Platonic Transformer is a group-equivariant transformer that uses Rotary Position Embeddings (RoPE) on the symmetry group of a Platonic solid. Key features:

- **Tetrahedral equivariance** (|G|=12): encodes 3D rotational symmetry via the chiral tetrahedral group
- **RoPE on values** (GTA Eq. 5): applies positional rotations to values and inverse-rotates the output
- **ScheduleFree optimizer** with r=2.0
- **O(3) data augmentation** (random rotations + reflections)
- **Charge/spin conditioning** (additive injection, following eSEN/UMA): random Fourier features over (charge, spin), Linear+SiLU mix, added to atom embedding
- **eSEN-aligned normalization**: target = (energy - element_refs) / 1.433569 (mean=0 enforced; rmsd is eSEN's published OMol-4M value)

See [ebekkers/platonic-scaling-laws](https://github.com/ebekkers/platonic-scaling-laws) for the scaling-laws analysis and [niazoys/PlatonicTransformers](https://github.com/niazoys/PlatonicTransformers) for the base PlatoFormer implementation.

See [PROGRESS.md](PROGRESS.md) for detailed progress tracking.

## Leaderboard

Reference baselines on OMol25-4M val (80 epochs, from AllScAIP paper Table 4):
- eSEN-md-d.       — E-MAE 1.32 meV/atom, F-MAE 6.78 meV/Å
- AllScAIP-md-d.   — E-MAE 1.04 meV/atom, F-MAE 8.19 meV/Å
- AllScAIP-md-ft-cons. — E-MAE 0.90 meV/atom, F-MAE 7.67 meV/Å (50 ep direct + 30 ep conservative fine-tune)

## wandb

Experiments tracked at [erikjbekkers/scaling-laws-symmetry](https://wandb.ai/erikjbekkers/scaling-laws-symmetry), group `platonic-omol` (or sub-groups per experiment series).

## Related

- [ebekkers/platonic-scaling-laws](https://github.com/ebekkers/platonic-scaling-laws) — Scaling-laws paper and experiments
- [niazoys/PlatonicTransformers](https://github.com/niazoys/PlatonicTransformers) — PlatoFormer base implementation
- [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) — FAIR Chemistry (OMol25 dataset, eSEN/UMA baselines)
- AllScAIP paper: arXiv:2603.06567 (FAIR, March 2026)
