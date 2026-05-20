# OMol25 training: Platonic Transformer + eSEN baseline

This document describes how to reproduce our **winning Platonic Transformer
recipe** (referred to as `qcczbpfn` in our internal logs) and the **eSEN
baseline** on the OMol25 force-field training task. Both runs use the
Lightning module in `mains/main_omol.py` and share the same dataloader,
loss, and metric pipeline.

## 1. Data

OMol25 is published by FAIR Chemistry on Hugging Face:
[https://huggingface.co/datasets/facebook/OMol25](https://huggingface.co/datasets/facebook/OMol25).

This PR's dataloader expects ASE LMDB files under:

```
${DATA_PATH}/open_mol/train_4M/  data0000.aselmdb ... data0079.aselmdb  (≈3.99 M molecules)
${DATA_PATH}/open_mol/val/       data0000.aselmdb ... data0079.aselmdb  (≈2.76 M molecules)
```

Set `dataset.data_dir` in the chosen config to `${DATA_PATH}`.

## 2. Environment

```bash
python -m venv venv
source venv/bin/activate

# Core install
pip install -r requirements.txt

# Optional: OMol25 extras (eSEN baseline only)
pip install fairchem-core==2.19

# Optional: flash attention (Hopper / Ada Lovelace)
pip install flash-attn==2.7.4
```

If `flash-attn` isn't available the model auto-falls back to the scatter
attention backend (see `attention_backend` in the Platonic config). The eSEN
baseline strictly needs `fairchem-core`.

## 3. Winning Platonic recipe (config `configs/omol.yaml`)

| Setting | Value |
|---|---|
| hidden_dim | 1920 |
| num_layers | 16 |
| num_heads | 12 (= |G|; head_dim = 160) |
| `solid_name` | tetrahedron |
| `qk_norm` | true |
| `swiglu` | false (GeLU-MLP FFN) |
| `activation` | "sin" (FFN) |
| `layer_scale_init_value` | 1e-4 |
| `rope_sigma` | 2.0 |
| `rope_on_values` | true |
| `use_key` | true |
| `chgspin_film` | true |
| `attention_backend` | flash |
| epochs | 20 |
| optimizer | AdamW, lr=5e-4, weight_decay=1e-8 |
| force loss weight `lambda_F` | 20.0 |
| EMA | 0.99 |

Reproduce:

```bash
python mains/main_omol.py --config configs/omol.yaml \
    --dataset.data_dir=/path/to/omol \
    --logging.enabled=true
```

Or via Slurm:

```bash
sbatch scripts/run_omol_platonic.sh
```

Expected throughput: ~370 ms/step on a single H100 at fixed `batch_size=32`;
20 epochs ≈ 34 h wall-clock. (Our internal `qcczbpfn` reached **~22 meV/atom
val MAE / ~22 meV/Å val force MAE** by ~130k steps, with continued
improvement; full 20-epoch curves available on request.)

> **Note on batch size**: the original winning recipe used a custom
> `DynamicBatchSampler` packing up to `max_atoms_per_batch=12000` per global
> step (~450 molecules/step). This PR ships fixed `batch_size=32` as a
> memory-safe default. To match the original effective batch size, port the
> `DynamicAtomBatchSamplerForAseDB` from the private `platonic-omol` repo.

## 4. eSEN baseline (config `configs/omol_esen.yaml`)

A conservative-force eSEN-small baseline, mirroring our internal `22630480`
reference run: sphere=hidden=32, lmax=4, mmax=2, 12 layers, autograd-derived
forces (`direct_forces=false`), activation_checkpointing on.

Reproduce:

```bash
python mains/main_omol.py --config configs/omol_esen.yaml \
    --dataset.data_dir=/path/to/omol
```

Or via Slurm:

```bash
sbatch scripts/run_omol_esen.sh
```

Expected throughput on H100, fp32: **~3,330 atoms/sec ≈ 16× slower per atom
than the Platonic recipe**. 20 epochs at `max_atoms_per_batch=2500` brushes
the 5-day partition limit on Snellius.

| Reference checkpoint | Val MAE (energy/atom) | Val MAE (force) | Steady-state ms/step |
|---|---|---|---|
| Our `22630480` @ step ~50k | ~31 meV | ~190 meV/Å | ~3,000 |

Reviewers can verify the **early loss curve** (steps 0–5,000) matches without
having to run to convergence — the smooth-energy recipe is highly
reproducible at the start of training.

## 5. Reproduction checklist

1. Fresh clone:
   ```bash
   git clone https://github.com/niazoys/PlatonicTransformers.git
   cd PlatonicTransformers
   ```
2. Install: `pip install -r requirements.txt` (+ optional extras as above).
3. Download OMol25 train_4M + val into `${DATA_PATH}/open_mol/`.
4. Edit `configs/omol.yaml` → `dataset.data_dir` (and likewise for the eSEN config).
5. Run the launcher of choice (`scripts/run_omol_platonic.sh` or `scripts/run_omol_esen.sh`).
6. Compare your wandb val curves to those linked in the PR description.

## 6. Architecture additions in this PR

Brief summary of what's new relative to upstream `niazoys/PlatonicTransformers`
main (commit `d356729`). Each is opt-in via config, default-off.

| Flag | Default | Purpose |
|---|---|---|
| `qk_norm` | false | RMSNorm on Q and K pre-RoPE (LLaMA-3 style). |
| `swiglu` | false | Gated FFN (`silu(W_gate(x)) * W_up(x) → linear2`). Pair with no LayerScale. |
| `activation: "sin"` | n/a | Adds `torch.sin` to the FFN activation registry. |
| `chgspin_film` | false | Per-block FiLM modulation `x ← (1+γ)·x + β` driven by per-graph charge/spin RFF + Linear+SiLU mix. Zero-init (identity at start). |
| `dataset.element_refs_path` | null | Per-element reference energy YAML; subtracted from target energy in train/val/test (flatter regression target). |
| `model.name: "esen"` | "platoformer" | Routes the Lightning module to `ESENModel` + `EquivariantNet` (eSCNMDBackbone). |
