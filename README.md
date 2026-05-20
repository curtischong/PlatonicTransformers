# Platonic Transformers — OMol25 production branch

> **Branch context.** This `production` branch of `ebekkers/platonic-omol`
> is a fork of the public Platonic Transformers paper code at
> [`niazoys/PlatonicTransformers`](https://github.com/niazoys/PlatonicTransformers)
> (tracking upstream `main` at commit `d356729`). The focus is **OMol25
> force-field training** — reproducing the `qcczbpfn` recipe (4× H100 DDP,
> 12k atoms/step effective batch) and running scaling experiments on top.
> The rest of the upstream codebase (CIFAR-10, QM9, ScanObjectNN, ImageNet
> mains + configs) is also present and runnable here; for those datasets
> see [upstream's README](https://github.com/niazoys/PlatonicTransformers/blob/main/README.md).
> This README only documents what is specific to this branch.
>
> The audit trail comparing this branch's output to `qcczbpfn` lives in
> [`HANDOVER.md`](HANDOVER.md).

---

## Quick start on Snellius (OMol25 force-field run)

```bash
git clone -b production git@github.com:ebekkers/platonic-omol.git
cd platonic-omol
source /scratch-shared/ebekkers/scaling-laws-venv-v2/bin/activate
./scripts/run_omol_snellius.sh 4     # 4× H100 (qcczbpfn-equivalent: 3000 atoms/rank × 4 = 12k effective)
./scripts/run_omol_snellius.sh 1     # 1× H100 (12000 atoms/step, same effective batch)
```

- **Data**: `/scratch-shared/ebekkers/omol25/open_mol/{train_4M,val}/` already exists. The `metadata.npz` next to each shard is the natoms-cache for dynamic batching — auto-detected. Only run `scripts/build_omol_natoms_cache.py <dir>` if you point at a different data dir.
- **Venv**: `/scratch-shared/ebekkers/scaling-laws-venv-v2` is pre-built (torch 2.8 / cu128, flash-attn 2 + 3, fairchem, lightning, ml-collections, mendeleev — all the deps `qcczbpfn` used). For a fresh env elsewhere, `./setup.sh` builds one from `requirements.txt`.
- **Recipe**: `configs/omol.yaml` is the canonical `qcczbpfn` recipe.
- **Overrides**: both launchers honor `DATA_PATH=...` and `VENV_PATH=...` env vars; edit `scripts/run_omol_platonic_snellius_{1,4}gpu.sh` for partition/wallclock.

---

## What this branch adds on top of upstream `niazoys/PlatonicTransformers@d356729`

All architecture additions are **opt-in via config**, default-off, so the
upstream paper recipes still reproduce byte-for-byte. The OMol-specific
modules are new files.

### Model / training architecture (opt-in flags in `configs/omol.yaml`)

| Flag | Default | What it does |
|---|---|---|
| `model.norm_type` | `"layernorm"` | Switch the block-level norm. `"rmsnorm"` swaps in an `RMSNorm` (with `quack` fused kernel on H100+ when available). |
| `model.qk_norm` | `false` | `RMSNorm` on Q and K pre-RoPE (LLaMA-3 style). Stabilises large-width attention. |
| `model.swiglu` | `false` | Replace the FFN `linear1→activation→linear2` with a gated MLP `silu(W_gate(x)) * W_up(x) → linear2`. |
| `model.activation` | `"gelu"` | FFN activation. Adds `"sin"` to the registry alongside `gelu` / `silu` / `relu` / `mish`. |
| `model.readout_activation` | `null` | Separate activation for the scalar readout MLP. `null` falls back to legacy `nn.GELU`. |
| `model.use_key` | `false` | When `true` + RoPE on, K projects through its own linear; when `false`, K is set to ones and RoPE provides the only Q–K signal. |
| `model.layer_scale_init_value` | `null` | LayerScale (CaiT) per-block γ init. `null` disables. Pairs naturally with `chgspin_film`. |
| `model.drop_path_rate` | `0.0` | Stochastic depth on each block's residual branch. |
| `model.chgspin_mode` | `"off"` | Per-graph charge/spin conditioning (eSEN/UMA recipe): `"off"` / `"add"` / `"concat"` injection of a Random-Fourier `Linear+SiLU`-mixed signal at the input. |
| `model.chgspin_film` | `false` | Per-block FiLM modulation `x ← (1+γ)·x + β` driven by the chgspin signal. Zero-init → identity at start. |
| `model.chgspin_layerwise[_gate]` | `false` | Alternative to FiLM: per-block additive injection of the chgspin signal, optionally gated. Mutually exclusive with `chgspin_film`. |
| `model.attention_backend` | `"scatter"` | `"scatter"` (default) / `"flash"` (FA2) / `"flash3"` (Hopper FA3, sm_90a only). |
| `model.qk_dim_factor` / `model.v_dim_factor` / `model.rope_v_independent` | `1, 1, false` | Asymmetric Q/K vs V head dims; independent V RoPE frequencies. Requires `flash`/`flash3`. |
| `model.interaction_radius` / `cutoff_p` / `max_num_neighbors` | `null, 6, 1000` | Sparse attention via `radius_graph(pos, r)` + Klicpera polynomial cutoff. Requires `dense_mode=false` + `attention_backend=scatter`. |
| `model.local_global` | `false` | Dual-stage local→global blocks (AllScAIP-style): each logical layer expands to a `(local, global)` `PlatonicBlock` pair. |
| `dataset.element_refs_path` | `null` | YAML of per-element reference energies (`omol_elem_refs`); subtracted from target energy in train/val/test for a flatter regression target. |

### OMol-specific files (new)

```
mains/main_omol.py                                    # Lightning module + main(); OMolModel + ESENModel
configs/omol.yaml                                     # canonical qcczbpfn recipe
configs/omol_esen.yaml                                # eSEN-small baseline
configs/constants/element_refs.yaml                   # per-element ref energies
platonic_transformers/datasets/omol.py                # OMolDataset, collate_fn, DynamicAtomBatchSamplerForAseDB
platonic_transformers/datasets/k_hot_encoding.py      # 92-d K-HOT atomic embeddings table
platonic_transformers/models/platoformer/force_field.py  # PlatonicForceField wrapper (atomic embedding + chgspin + transformer + fp64 energy reduction)
platonic_transformers/models/platoformer/chg_spin_emb.py # Random-Fourier ChgSpinEmbedding
platonic_transformers/models/baseline/esen/           # conservative-force eSEN baseline (eSCNMDBackbone + MLP_EFS_Head)
platonic_transformers/utils/callbacks.py              # EMACallback (swaps in EMA weights at val/test) + Memory/Timer callbacks
scripts/run_omol_snellius.sh                          # Snellius dispatcher (1 or 4 H100)
scripts/run_omol_platonic_snellius_{1,4}gpu.sh        # underlying launchers
scripts/run_omol_platonic_hipster.sh                  # hipster 4× RTX 6000 Ada launcher
scripts/run_omol_platonic.sh                          # generic launcher
scripts/build_omol_natoms_cache.py                    # builds metadata.npz natoms cache for custom data dirs
```

---

## OMol25 recipe (`configs/omol.yaml` = qcczbpfn)

| Setting | Value |
|---|---|
| `hidden_dim` | 1920 |
| `num_layers` | 16 |
| `num_heads` | 12 (= \|G\|; head\_dim = 160) |
| `solid_name` | `tetrahedron` |
| `dense_mode` | false (sparse / graph mode) |
| `attention_backend` | `flash` |
| `qk_norm` | true |
| `use_key` | true |
| `swiglu` | false (GeLU-MLP FFN) |
| `activation` | `"sin"` |
| `layer_scale_init_value` | 1e-4 |
| `rope_sigma` | 2.0 |
| `rope_on_values` | true |
| `chgspin_mode` | `"add"` |
| `chgspin_film` | true |
| `chgspin_mix_init_std` | 0.02 |
| `train_augm_group` | `"o3"` (rotation + reflection) |
| `epochs` | 20 |
| `optimizer` | AdamW, lr=5e-4, weight\_decay=1e-8 |
| `lambda_E` / `lambda_F` | 10 / 20 |
| `scheduler` | `cosine_annealing_ws`, 1% warmup |
| `EMA` | 0.99 (warmup 2000 steps) |
| `compile` | true (`mode=default`) |
| effective batch | 12 000 atoms / 2.4 M edges per optimizer step (4× DDP × 3000/600k per rank) |

The pre-computed `metadata.npz` next to each `.aselmdb` shard is required
when `training.dynamic_batching=true` (the recipe default); the dataset
loader raises a clear error if it's missing.

---

## Repository layout

```
.
├── HANDOVER.md                  # audit trail vs qcczbpfn (read this before iterating)
├── archive/                     # frozen private platonic-omol training framework (reference)
├── baseline-reference/          # local qcczbpfn wandb dump + metadata.npz (gitignored)
├── configs/                     # all yaml configs (omol*, cifar10, qm9*, scanobjectnn, imagenet)
├── mains/                       # one file per dataset (main_omol.py is the OMol entry point)
├── platonic_transformers/
│   ├── datasets/                # dataset loaders (omol.py, qm9.py, scanobjectnn.py, ...)
│   ├── models/
│   │   ├── platoformer/         # PlatonicTransformer, blocks, conv, linear, RoPE, force_field.py
│   │   └── baseline/esen/       # eSEN baseline (used by OMolModel.name=esen)
│   └── utils/                   # CosineWarmupScheduler, EMACallback, config_loader
├── scripts/                     # SLURM launchers (run_omol_snellius.sh is the entry point)
├── meta_main.py                 # legacy upstream dispatcher (works for non-OMol mains)
├── setup.sh                     # uv-based env builder (uses requirements.txt)
└── pyproject.toml / requirements.txt
```

---

## Citation

If this code is useful for your research, please cite the Platonic
Transformers paper:

```bibtex
@misc{islam2025platonictransformerssolidchoice,
      title={Platonic Transformers: A Solid Choice For Equivariance},
      author={Mohammad Mohaiminul Islam and Rishabh Anand and David R. Wessels and Friso de Kruiff and Thijs P. Kuipers and Rex Ying and Clara I. Sánchez and Sharvaree Vadgama and Georg Bökman and Erik J. Bekkers},
      year={2025},
      eprint={2510.03511},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.03511},
}
```

MIT-licensed; see [`LICENSE`](LICENSE) (when present in upstream) or
upstream's repo.
