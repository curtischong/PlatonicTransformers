# Opening prompt for next session — Platonic OMol25 leaderboard

Copy-paste this to start a fresh session:

---

You are helping with the **Platonic OMol25 leaderboard** project.

**Important:** If you are working from a Nomi workspace, the project repo is nested at `platonic-omol/platonic-omol/`. All paths below are relative to the inner repo root.

**Scope:** Search for the best PlatoFormer baseline + fair comparison against the new eSEN baseline. Snellius H100 is the primary cluster. Do NOT touch the platonic-scaling-laws repo / GCP v9 sweep — separate workspace.

**Read first (in order):**
1. `../README.md` (the workspace README) — top section "2026-05-11" covers what was added and the throughput numbers.
2. `PROGRESS.md` — historical context.
3. `README.md` (inner repo) — project overview.

## Handover summary (2026-05-11, end of day)

### Branch
`precision-experiments-2026-05-08` (name predates the new work — kept for continuity; rename only if it becomes confusing). Last commit `05ecd31` "esen: fairchem eSCN-MD wrapper + ns/day throughput benchmark".

### Currently in flight on Snellius (check first via `ssh snellius 'squeue -u ebekkers'`)

| Job | W&B run | What | Wall in |
|---|---|---|---|
| `22616965` | (wd-sweep group) | PT rs=4.0 + EMA=0.99, 20ep | started 2026-05-10 |
| `22620086` | (wd-sweep group) | PT rs=1.0 + EMA=0.99, 20ep | started 2026-05-10 |
| `22628257` | (wd-sweep group) | **PT rs=2.0 + EMA=0.99, 80ep** | started 2026-05-11 ~09:00 UTC |
| `22630480` | `7ppvskcg` esen-baseline-20ep | **eSEN-small 20ep** | started 2026-05-11 ~07:45 UTC |

All four are 1×H100. **eSEN at 5-day cap will get ~14 epochs of 20** (≈7.4 day projected; partition limit 5d). 80-epoch PT projected ~4 days, fits the cap with ~24h slack.

W&B project: `omol-leaderboard/scaling-laws-symmetry`. The PT recipe being searched: h1920, l=8, ffn_dim_factor=2, sin/sin activations, layer_scale=1e-4, rope_on_values=true, chgspin_mode=add, fp32 baseline. The "winning small recipe so far" is `77j0ulg4` (rs=2.0 + EMA=0.99 + 20ep) — that's what the 80-epoch run is extending.

### What was built on 2026-05-11

1. **eSEN baseline trainable in our pipeline.** New wrapper `training/nets/uma/model.py` (`EquivariantNet`) around fairchem's `eSCNMDBackbone` + `MLP_EFS_Head` with `direct_forces=False` so forces are conservative (autograd from energy — the "smooth-energy" recipe that distinguishes eSEN from UMA-direct). Config `training/configs/force_field_module/esen.yaml` (3.4M params: sphere=hidden=32, lmax=4, mmax=2, 12 layers, activation_checkpointing=true). Launch `scripts/run_esen_small_20ep_fp32.sh`. Mandatory: `trainer.inference_mode=false` for autograd-grad val.

2. **ns/day inference benchmark** matching AllScAIP (Qu et al. 2026, arXiv:2603.06567) Table 2 protocol. `training/benchmark_ns_per_day.py` + `scripts/run_benchmark_ns_per_day.sh`. `MODE=single` constructs one fairchem `AtomicData` system of N atoms (random C/H/N/O, uniform in 30Å box) → paper protocol. `MODE=batched` uses one real dynamic-batched mini-batch. Both single-GPU, fp32, forward only, dt=1 fs.

### Headline result (single H100, N=1000 atoms, single-system mode, fp32 forward)

| Model | Params | ms/step | **ns/day** | atom-ns/day |
|---|---|---|---|---|
| PT (`77j0ulg4` recipe) | 18.2M | 24.7 | **3.49** | 3.5e3 |
| eSEN-small | 3.4M | 140.0 | **0.62** | 6.2e2 |

H100→H200 translation: ~0% for attention models, ~+30% for eSCN/eSEN. So our PT ≈ 3.49 H200-equivalent. Vs paper's full-encoding numbers on H200: AllScAIP-sm 35M = 2.279, AllScAIP-md 85M = 1.124. **Our 18M PT runs ~1.5× faster than AllScAIP-sm and ~3× faster than AllScAIP-md at much smaller size.**

The eSEN-small lag vs the paper's eSEN-sm (~5 ns/day, 6M) is **architectural**, not implementation: our config has lmax=4 (paper lmax=2 → 2.7× SH-coefficient cost), 12 layers (paper ~6), and otf_graph=on (paper precomputes).

## To explore next

### 1. Multi-GPU PT training (revisit)
A previous session benchmarked 2-GPU DDP at ~1.68× throughput vs 1-GPU on the sig2 recipe (~3.1h/epoch vs ~5.2h). Worth re-running on the current 1920d/sin/EMA recipe and pushing to 4-GPU. Setup recap:
- Submit with `--gres=gpu:h100:2 --ntasks=1 --cpus-per-task=32 --mem=300G` (single SLURM task — Lightning forks rank 1 itself).
- `export SLURM_JOB_NAME=bash` before `python` (disables Lightning's SLURM auto-detect).
- `trainer.devices=N trainer.strategy=ddp` via Hydra.
- `src/model/omol_module.py` already logs `token_processed`/`total_flops_used` with `sync_dist=True, reduce_fx="sum"` so multi-GPU reports global atoms correctly.
- The custom batch sampler (`DynamicAtomBatchSamplerForAseDB`) shards itself via `dist.get_rank()/get_world_size()`; set `+trainer.use_distributed_sampler=false` to skip Lightning's auto-sampler injection.
- A 4-GPU run on hipster died at 1440G mem-request — node ceiling is 770G; cap at 720G if revisited there.

### 2. Fair comparison vs eSEN
Today's comparison was forward-only inference at N=1000 (apples-to-apples for throughput). For a fair *accuracy* comparison we still need to decide what's "fair":
- **Equal params**: scale eSEN up (more channels/layers — likely beyond our 18M PT target) OR scale PT down. Pick one direction.
- **Equal batch size (atoms/step)**: PT is at max_atoms=12000, eSEN at 2500 (full 12000 OOMs at lmax=4). To match, either drop PT to max_atoms=2500, or test whether eSEN-small can run at 4000–6000 with activation_checkpointing already on. 5× more steps/epoch at smaller batch will stress the LR schedule — likely needs re-tuned warmup fraction.
- **Equal compute budget (FLOPs or wall-clock)**: maybe the most defensible. Run eSEN to whatever epoch fits in PT's per-day wall, and report at matched FLOPs.

A cleaner eSEN reference matching the paper's "eSEN-sm" recipe (lmax=2, ~6 layers, hidden=128) would also clarify whether the gap is recipe or architecture. Worth building a second config `esen_sm_paper.yaml` and benching.

### 3. Other items pulled forward
- Precision-experiment edits in `nets/platoformer/{model,platoformer}.py` are uncommitted on this branch. Decide: land them, revert, or move to a separate branch. They're noise on `git status -s`.
- `NEXT_SESSION_PROMPT.md` (this file) and `PROGRESS.md` are modified-uncommitted historical notes. Cleanup pass advisable.

### 4. Leaderboard submission pipeline (still untouched)
- Figure out npz format for the [FAIR Chemistry Leaderboard](https://huggingface.co/spaces/facebook/fairchem_leaderboard).
- `test_omol.py` already exists as the full-val entrypoint.
- Inference script to generate predictions on OMol25 test split + submit to the HF space.

## How to check status quickly

```bash
# Queue + wall
ssh snellius 'squeue -u ebekkers --format="%.10i %.40j %.8T %.10M %.10L"'

# Pull recent train_loss / step count from W&B for a specific run
ssh snellius 'source /scratch-shared/ebekkers/scaling-laws-venv-v2/bin/activate && python3 -c "
import wandb; api=wandb.Api(); r=api.run(\"omol-leaderboard/scaling-laws-symmetry/<RUNID>\")
print(r.name, r.state, r.summary.get(\"_runtime\",0), r.summary.get(\"trainer/global_step\",0))"'

# Run the throughput benchmark on a fresh PT or eSEN config
ssh snellius 'cd /scratch-shared/ebekkers/platonic-omol &&
  sbatch --job-name=bench-pt2-single  --export=ALL,MODEL=platoformer,MODE=single,N_ATOMS=1000 scripts/run_benchmark_ns_per_day.sh'
```

**Ask me what we're working on today before starting any task.**
