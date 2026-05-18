# Platonic OMol25 — Progress Tracker

## Goal
Get the Platonic Transformer onto the [FAIR Chemistry Leaderboard](https://huggingface.co/spaces/facebook/fairchem_leaderboard) for OMol25.

## Current status (2026-05-07) — fp32 baseline cancelled, bf16 speed comparison launched

**Decision.** The Snellius `22430304 PT2-sig4-wd1e4-dyn` run was cancelled at epoch 29 / 4d 7h (out of planned 100 epochs / 5d wall) — diminishing returns on the val curve, and the H100 slot was better spent on a hardware-optimisation comparison. Last checkpoint: `epoch=29.ckpt` / `last.ckpt` at `…/run_22430304_params_33.7 million/`, May 7 10:51.

### Two follow-ups submitted

1. **Full-val test on the cancelled checkpoint** (job `22563360 PT2-sig4-wd1e4-test`, 1× H100, 2h wall, expected ~25–30 min). Test-only entrypoint `training/test_omol.py` (new, see below). Evaluates `last.ckpt` on the **full** `open_mol/val` (2.76M heldout) — i.e. drops the `+trainer.limit_val_batches=500` cap from training. Logs to W&B as `pt2-upstream-add-sig4-wd1e4-dyn-fullval` in group `pt2-upstream-sig4-wd-snellius`. *Number not yet landed at session close.*
2. **bf16-mixed + TF32 speed comparison** (job `22562808 PT2-sig4-wd1e4-bf16`, 1× H100, 5d wall, started 2026-05-07 ~16:55). Identical recipe to the cancelled fp32 baseline (sig=4, wd=1e-4, dyn-batch, no EMA, seed=1) **except** `+trainer.precision=bf16-mixed` and `export PSL_FAST_BACKENDS=1` (cudnn benchmark + TF32 high). Different `EXP_NAME=…-bf16` so checkpoints don't collide. Same W&B group for overlay.

### Speed audit (the *why* for the bf16 run)

While picking the optimisations to try:
- **Original recipe was pure fp32 at the Lightning Trainer level** — no `precision=` override anywhere in configs or launch scripts. On H100 that's ~2–4× slower than bf16 for the matmul-bound layers.
- **TF32 was gated behind `PSL_FAST_BACKENDS=1`** which the existing launch scripts never set, so `cudnn.deterministic=True` was active and `set_float32_matmul_precision` defaulted (TF32-off-ish).
- **Forces are predicted directly** by `self.net(data)` (no autograd-through-energy), so bf16 mixed-precision is straightforward — no precision concerns on the gradient computation side.
- FlashAttention path requires fp16/bf16 anyway, so internal casting was already happening.
- **`head_dim=48`** (1728 / 36) is FA2's least-favourite size — pads to 64 internally → ~25% lost attention throughput. Fix would be `nhead=27` (head_dim=64), but that's a model change requiring a fresh train.
- Other untried wins for later sessions: FA3 on Hopper (TMA + async), `torch.compile(mode="max-autotune")` (set `PSL_COMPILE_MODE=max-autotune`), DDP `static_graph=True`, audit `sync_dist=True` log calls.

**How to compare once both runs have data:** look at ms/step at step ~5000–7500 on W&B. Original sig4-wd1e4 was ~334 ms/step at 1× H100; expect bf16 + TF32 to land ~150–220 ms/step.

### Test-only entrypoint (new, two bugs found and fixed)

`training/test_omol.py` — ~30-line script that loads a checkpoint and calls `trainer.test()` with no `trainer.fit()`. While building it I hit two latent issues with the test-only path:
1. `trainer.test(dataloaders=…)` does **not** auto-call `datamodule.setup()`, so `self.datasets["val"]` was missing → `KeyError: 'val'`. Fix: explicit `datamodule.setup(stage="test")` before `val_dataloader()`.
2. `omol_module.py` initialises `self.free_scheduler` inside `configure_optimizers()`, which Lightning skips for test-only. `on_test_model_eval()` then crashes on `AttributeError: 'GraphModel' object has no attribute 'free_scheduler'`. Fix: set `model.free_scheduler = False` after instantiation in `test_omol.py` (test path has no optimiser, so False is correct).

Both fixes live in `test_omol.py` only — `omol_module.py` is untouched.

### Files added this session (uncommitted on laptop and Snellius working tree)

- `training/test_omol.py` — test-only entrypoint
- `scripts/run_test_sig4_wd1e4_dyn_snellius.sh` — sbatch wrapper for full-val test
- `scripts/run_pt2_upstream_long_sig4_wd1e4_dyn_bf16.sh` — bf16 + TF32 variant of the long training script

---

## Current status (2026-05-05) — sig4 + WD/EMA exploration

**Recipe direction.** The best of the original sigma sweep was **`its7gzf1`** on Snellius (sig=1.5, dyn-batching, 1× H100, no EMA, no WD). Since then the variation knob has been **rope_sigma=4.0** (vs 1.5 default) — most active runs use it. Open questions being probed: (a) does weight decay help; (b) does EMA help; (c) what's the right effective batch size.

### Active runs (across 3 clusters)

| Cluster | Job | Recipe | wandb |
|---|---|---|---|
| Snellius `gpu_h100` | `22430304 PT2-sig4-wd1e4-dyn` | sigma=4, **wd=1e-4**, 1×H100, dyn max_atoms=12000 | [jtvde113](https://wandb.ai/omol-leaderboard/scaling-laws-symmetry/runs/jtvde113) |
| ivi `geodude` (A5000) | `161828 PT2-sig4-dyn-ivi-geodude` | sigma=4, **wd=0**, no EMA, scatter, dyn max_atoms=4000 | [kfyw7vv9](https://wandb.ai/omol-leaderboard/scaling-laws-symmetry/runs/kfyw7vv9) |
| hipster `performance` | `249334 PT2-sig4-wd1e2-hipster` | sigma=4, **wd=1e-2**, scatter, dyn max_atoms=12000 | [4znkxhfw](https://wandb.ai/omol-leaderboard/scaling-laws-symmetry/runs/4znkxhfw) |
| hipster `performance` | `249335 PT2-sig4-wd1e3-hipster` | sigma=4, **wd=1e-3**, scatter, dyn max_atoms=12000 | [5zikayhj](https://wandb.ai/omol-leaderboard/scaling-laws-symmetry/runs/5zikayhj) |
| ivi `all6000` (Turing) | `163575 PT2-sig4-wd05-bs32-all6000` | sigma=4, **wd=0.05, fixed bs=32**, scatter | (just submitted, pending) |
| ivi `all6000` (Turing) | `163576 PT2-sig4-wd05-bs64-all6000` | sigma=4, **wd=0.05, fixed bs=64**, scatter | (just submitted, pending) |

### Recently dead
- `22430305` snellius 2-GPU wd=1e-4 — host OOM-killer at epoch 18 (val 0.376, was healthiest run). Checkpoint exists.
- `161829/30/31` ivi EMA-sweep (decay=0.9999/0.999/0.99999) — slurm `KeyboardInterrupt` 2-4d in, no Python exception. Ema=0.99999 was overfitting; the other two were trending well (val 0.32-0.35 at epoch 4). Checkpoints exist.
- hipster `wd=1e-4/1e-5/1e-6` — user-cancelled in favour of focusing on the larger WD values.

### Cross-cluster gotchas (codified in the agent's memory; here for humans)
- **flash-attn 2.7+ dropped Turing AND ignores `TORCH_CUDA_ARCH_LIST`** — the build hardcodes sm_80/90/100/120 only, no PTX, so the wheel won't even JIT to sm_86 (A5000) or sm_75 (Turing). On ivi (A5000/Turing) and hipster (Ada sm_89) the long runs use **`attention_backend=scatter`** (graph_scattered_attention via _scatter_softmax) — semantically equivalent to flash, slower per step.
- **Snellius `scaling-laws-venv` was found wholesale corrupted (~103 packages with empty RECORD files)** on 2026-05-03. Built fresh **`/scratch-shared/ebekkers/scaling-laws-venv-v2`** — this is what the new launch scripts (`run_pt2_upstream_long_sig4_*_dyn.sh`, `_2gpu`) point to. The original venv stays in place for the still-running V11-PT-F job.
- **Snellius working tree `/scratch-shared/ebekkers/platonic-omol/`** had `.py` files deleted in `training/utils/`, `training/nets/`, `training/src/normalization/`, etc. (same `.py`-gone-but-`.pyc`-survived pattern as the venv). Restored from `origin/upstream-port-pt2`. If a new run fails on `ModuleNotFoundError: No module named 'utils.log'` etc., this is the cause — `git checkout origin/upstream-port-pt2 -- training/`.
- **ivi `geodude`** has a per-account concurrent jobs cap of **4** (`grpjobs=4` on `geodudeusers`).
- **ivi `all6000`** requires **`--account=all6000users`** (default `linuxusers` is blocked, slurm parks the job indefinitely with reason `(PartitionConfig)` + sliding `START_TIME`).
- **ivi torch.compile** needs **`CUDA_HOME=/usr/local/cuda-12.9`** — the default `/etc/alternatives/cuda` resolves to 13.1 which torch (cu128) can't use; inductor errors out as `PermissionError: 'nvcc'` because the symlinked nvcc isn't usable. All ivi launch scripts set this.
- **hipster `performance`** has 5 nodes; was in `DRAIN+REBOOT` for ~5 days late April / early May. Now back to normal. CUDA on hipster: `/usr/local/cuda-12.3` is what's available (`12.9` doesn't exist there).
- **e_mae per atom**: `omol_module.py:_compute_loss` logs both `e_mae` (per-molecule meV, legacy) and `e_mae_per_atom` (meV/atom, the leaderboard convention). Use `e_mae_per_atom` for any direct comparison to AllScAIP / eSEN.
- **EMA**: `EMACallback` lives in `training/src/utils/callbacks.py` and activates when `+ema.decay=...` is set as a Hydra override. Default is no EMA.

### Obvious next steps if resumed
1. **Once the wd=0.05 fixed-batch pair (163575, 163576) start** — see whether the heavier WD + smaller batch helps stability vs the 1e-2/1e-3 dyn-batching hipster runs.
2. **Pick a winner** of the WD sweep (compare on tokens, not steps) — currently runs differ on cluster + GPU + batch, so the cleanest cross-comparison is `4znkxhfw` (1e-2) vs `5zikayhj` (1e-3) on the same hipster setup.
3. **Resume from checkpoint** the snellius 2-GPU wd=1e-4 run (it was the healthiest, just OOM'd). Lower `--mem` slightly or reduce dataloader workers; the model itself was fine.
4. **Decide** whether to continue probing EMA on OMol scale — the ivi EMA runs all died from slurm, not from the model. Could resubmit on hipster which has more wallclock headroom now.
5. **Per-layer charge/spin injection** (eSEN-style) and other AllScAIP tricks (ERoPE, LAE, ffn_multiplier=2) are still untried.

### Original 2026-04-18 sigma sweep (older)

### Active Snellius jobs
- **21937571** (`omol-4m-sig2`): PT-2, 4M, 80 epochs, bs=64, **sigma=2.0**, 1× H100. `val_check_interval=5000`, `limit_val_batches=500`, val on 10%-of-train split (`validation_mode=train_split` default). 5-day wall. Running since 2026-04-16 20:35.
- **21943002** (`omol-4m-sig15-long`): PT-2, 4M, 80 epochs, bs=64, **sigma=1.5**, 1× H100. Same eval config as sig2. Running since 2026-04-17 02:40.
- **21951463** (`omol-4m-sig2-2gpu`): PT-2, 4M, 80 epochs, bs=64/rank, **sigma=2.0, 2× H100 DDP** (effective batch 128). `val_check_interval=2500`, `limit_val_batches=250` so val curves align with 1-GPU sig2 on samples-processed axis. Running since 2026-04-17 11:14.
- **21965445** (`omol-4m-sig4`): PT-2, 4M, 80 epochs, bs=64, **sigma=4.0**, 1× H100. Started 2026-04-17 ~20:15.
- **21965446** (`omol-4m-sig8`): PT-2, 4M, 80 epochs, bs=64, **sigma=8.0**, 1× H100. Started 2026-04-17 ~20:15.

### 2-GPU DDP test (2026-04-17)

First 2-GPU run (21951463) works end-to-end. Path to working config:
1. `srun --ntasks=2` on Snellius doesn't forward GRES to child tasks → torch.cuda empty, job failed. Replaced with `ntasks=1 + python` (no srun) so Lightning handles process spawning.
2. Lightning auto-detected SLURM and deferred launch to srun-that-wasn't → world_size=1. Fix: `export SLURM_JOB_NAME=bash` to disable Lightning's SLURM detection, falling back to its subprocess launcher.
3. For val curves to overlap 1-GPU runs on the samples-processed axis: `limit_val_batches=250` (combined strided 32K subset matches 1-GPU's 0..31999), `val_check_interval=2500` (val every 320K samples, same cadence), and fix `token_processed` / `total_flops_used` logs with `sync_dist=True, reduce_fx="sum"` so DDP reports the global atom count.

Measured throughput (~8h into 2-GPU run, steps 5000→7500 interval):
- 1-GPU sig2: ~334 ms/step avg, ~192 samples/sec
- 2-GPU sig2: ~396 ms/step, ~323 samples/sec
- **Speedup: ~1.68×** (not the ideal 2×; NCCL all-reduce + sync_dist reductions eat ~15–20%)
- Estimated per-epoch: ~3.1h on 2-GPU vs ~5.2h on 1-GPU

### Key finding — val-loss comparisons across runs

When comparing runs with `+trainer.limit_val_batches=500`, remember:
1. **The val loader has `shuffle=False`**, so `limit_val_batches=500` always evaluates on the exact same first 500 batches (32K samples) — deterministic within and across runs.
2. Different runs may point at different `val_data_path` (e.g. `open_mol/val` = 2.76M official heldout vs `open_mol/neutral_val` = ~400K neutral-only). The first 500 batches of each are fixed but different molecules. **Losses across different val_data_paths are NOT comparable.**
3. Under DDP with default Lightning `DistributedSampler`, each rank runs its own `limit_val_batches` on strided indices — combined val set is 2× larger than 1-GPU. To keep parity: halve `limit_val_batches` on 2-GPU runs (see sig2-2gpu recipe).
4. For the final leaderboard eval, remove `limit_val_batches` and run on full val.

Observation: the ~0.047 val-loss offset between the cancelled v3 (0.074) and sig15-long (0.121) at matched steps was entirely due to val set difference (`val` vs `neutral_val`); train-loss trajectories agree to ~7-8% (not bit-exact — attributed to torch.compile + O(3) augmentation RNG noise, same recipe otherwise).

### Cancelled earlier (2026-04-17)
- `21932720` (`omol-4m-v3`, sig=1.5, 13 epoch, official heldout val): superseded by the 80-epoch long run.
- `21943030` (duplicate sig=1.5 long): removed to avoid running two identical jobs in parallel.
- `21950203`, `21950429`, `21950665` (2-GPU DDP attempts): first two failed with GRES/DDP issues, third ran at world_size=1. All superseded by 21951463.

### Infrastructure
- [x] Repo: [ebekkers/platonic-omol](https://github.com/ebekkers/platonic-omol) (private)
- [x] Training code from platonic-scaling-laws with charge/spin support
- [x] Charge/spin: additive injection following eSEN/UMA (embed → Linear → SiLU → add to atom features, once at input)
- [x] 4M data module with charge/spin from `atoms.info`, precomputed reference energies
- [x] Snellius: repo at `/scratch-shared/ebekkers/platonic-omol`, data at `/scratch-shared/ebekkers/omol25/open_mol/`
- [x] Official val split extracted: `val/` (2.76M samples, 80 aselmdb files, 23 GB)

### Data on Snellius (`/scratch-shared/ebekkers/omol25/open_mol/`)
- `neutral_train/` — 34M neutral structures
- `neutral_val/` — neutral validation (~28K samples)
- `train_4M/` — 4M subset (53 aselmdb files, 14 GB)
- `val/` — official heldout validation (2.76M samples, 23 GB) — use `limit_val_batches=500` during training

### Lessons learned
- **Dynamic batching + torch.compile = recompilation hell.** Variable batch sizes cause torch.compile to hit recompile limit. Use fixed `batch_size` with `dynamic_batching=false`.
- **Val set is huge (2.76M).** Full validation takes ~3.5h. Use `+trainer.limit_val_batches=500` during training, full val only at the end.
- **bs=128 OOMs** on rare batches with many large molecules (4M split has molecules up to 350 atoms). Stick to bs=64.
- **validation_mode=heldout** uses the official val split. `train_split` wastes training data and creates huge val sets.
- **`+` prefix needed** for `trainer.limit_val_batches` and `force_field_module.net.rope_on_values` and `force_field_module.optimizer.r` (not in base Hydra config).

### Architecture decisions
- **Charge/spin:** Additive injection (eSEN/UMA style). NOT AdaLN (removed), NOT concatenation.
- **eSEN adds charge/spin at every layer** (per the OMol25 paper). We currently add once at input. Consider per-layer injection as a future improvement.
- **ScheduleFree optimizer** (r=2.0) — no schedule to tune, works well for multi-epoch.

### Recipe
| Setting | Value |
|---------|-------|
| hidden_dim | 1728 |
| nhead | 36 |
| num_layers | 12 |
| solid_name | tetrahedron |
| dense_mode | false |
| layer_scale_init_value | 1e-4 |
| rope_on_values | true |
| rope_sigma | 1.5 |
| train_augmentation | o3 |
| optimizer | ScheduleFree (r=2.0) |
| lr | 5e-4 |
| compile | true |
| batch_size | 64 (fixed, no dynamic batching) |
| charge_emb_dim | 64 |
| spin_emb_dim | 64 |
| val_check_interval | 1000 |
| limit_val_batches | 500 |

### Estimated epoch times (H100)
- 4M split: ~2.2h per epoch (+ ~1h val overhead = ~3.2h total)
- Neutral 34M: ~18.5h per epoch

## Next steps
1. **Monitor** the 5 running jobs. Watch for 5-day wall-time expiry; resubmission may be needed to continue past that.
2. **Compare the rope_sigma sweep** (1.5, 2.0, 4.0, 8.0) once sig4/sig8 reach step ~70K. sig2-2gpu should overlap sig2 on the samples-processed axis — good alignment test for DDP correctness.
3. **Leaderboard submission:** Figure out npz format for HuggingFace leaderboard.
4. **Inference script:** Generate predictions on OMol25 test splits.
5. **Per-layer charge/spin injection** (matching eSEN) — potential improvement.
6. **Full validation** at end of training (remove limit_val_batches for final eval).

## Deferred experiments — queued until PlatonicTransformers PR lands
Gated on the `platonic-repo-update` PR being finalized. Once that's done, run on the OMol 4M setup using the current best PT-2 recipe as baseline:

1. **EMA impact on OMol 4M.** OMol training currently has no EMA. Port `EMACallback` from repo-update (`platonic_transformers/utils/callbacks.py`) and run a baseline-vs-light-EMA pair (decay=0.99). Note: QM9 regression dropped EMA (`f5473c4`) after finding it unhelpful, but this hasn't been tested on OMol's longer/larger regime.
2. **Symmetric 2-layer readouts.** Test `vector_readout` shrunk to 2 layers (matching `scalar_readout`). The repo-update branch tried this (`eabe62f`) and reverted (`9614c8e`) based on QM9-gen training-spike forensics — the verdict was about gen-training divergence, not OMol regression performance.
3. **AdaLN vs concat for charge/spin conditioning.** OMol currently uses AdaLN modulation (`chg_spin_emb.py` + `conditioning_dim` path in `block.py`). Compare against the simpler concat-onto-scalar-channels approach that QM9 gen settled on (`9614c8e` revert rationale: "Noise conditioning = concat c_noise onto scalar input channels … 0 extra modules").

## Reference baselines
| Model | Split | Epochs | F-MAE (meV/Å) |
|-------|-------|--------|---------------|
| eSEN-30M | full | 12 | — |
| eSEN-30M | 4M | 80 | 10.1 |
| PlatoFormer PT-2 (ours, v9) | neutral | 1 | ~14.1 |
