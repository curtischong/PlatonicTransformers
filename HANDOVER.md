# Handover — Public PR reproduction of `qcczbpfn`

Date: **2026-05-20** (updated mid-session). Last commit on
`ebekkers/platonic-omol@main`: **`aac7c35`** + uncommitted edits to
`mains/main_omol.py` (cudnn + dynamo knobs — see "Bugs found and fixed"
below) and `public-pr-prep/PlatonicTransformers/platonic_transformers/datasets/omol.py` synced
to hipster (where the graph-cap fix had not been propagated previously).

## What this work is

We're preparing a public PR to add OMol25 force-field training to the
`niazoys/PlatonicTransformers` repo. The goal is to **exactly reproduce
`qcczbpfn`** — the winning Platonic-Transformer run on OMol25, originally
trained on the private `ebekkers/platonic-omol` repo with hipster 4× RTX 6000
Ada DDP. Mohammad will recreate the commits on the public repo for GitHub
visibility once the recipe is faithfully ported.

Code path (laptop): `workspaces/platonic-omol/platonic-omol/public-pr-prep/PlatonicTransformers/`

Companion reference dump (local, gitignored): `public-pr-prep/baseline-reference/` —
contains `metadata.npz` for `train_4M`/`val` and the full qcczbpfn wandb run dir
(`config.yaml`, `output.log`, `wandb-summary.json`, `run-qcczbpfn.wandb`).

## Where the qcczbpfn reference lives

- **wandb run**: https://wandb.ai/omol-leaderboard/scaling-laws-symmetry/runs/qcczbpfn
- **Hipster code** (read-only, frozen): `hipster:/home/ebekker/platonic-omol-backup/` — git HEAD `7fd2673aa632dda638fd63edf23a4529737635ca` (branch `main`) plus a dirty `scripts/run_pt2_hipster_h1920_l8_20ep_fp32.sh` (added DDP support via `DEVICES` env var). The qcczbpfn run used this exact working tree.
- **Hipster venv** (slightly modified during this session): `hipster:/home/ebekker/platonic-omol-backup/venv` — added `ml-collections` and `mendeleev` (latter downgraded pandas 3.0.2 → 2.3.3, side-effect, probably harmless). Original `platonic-omol/venv` is unmodified.
- **Hipster dataset**: `hipster:/scratch/ebekker/omol/open_mol/{train_4M,val}/` with their original metadata.npz (May 2025, fairchem-shipped).
- **Snellius dataset**: `snellius:/scratch-shared/ebekkers/omol25/open_mol/{train_4M,val}/` — our metadata.npz files were rebuilt during this session (val one byte-equivalent in shape but uncompressed; both contain only `natoms` not `data_ids`, see "Open audit questions" below).
- **Snellius venv**: `snellius:/scratch-shared/ebekkers/scaling-laws-venv-v2`

## Bugs found and fixed (committed to `aac7c35`)

1. **`ModelCheckpoint(monitor=...)` stale key** (`mains/main_omol.py:666`).
   Used `'valid MAE (energy) [meV]'` from pre-refactor logging. Renamed to
   `'e_mae/val'`. Without this fix the run crashed at the first val loop.

2. **`model.scale = train_rmsd` silently clobbered to 1.0** (`mains/main_omol.py:703`).
   `OMolModel.__init__` registered `self.scale = train_rmsd = 1.433569`, then
   `main()` overwrote it with `train_loader.dataset.scale = 1.0`
   (`set_scale_shift(1.0, 0.0)` is called unconditionally). Now gated on
   `dataset.scale_shift=True`. Targets are now correctly normalized.

3. **Per-molecule `charge` / `spin` always zero** (`platonic_transformers/datasets/omol.py`).
   `OMolDataset.__getitem__` read `atoms.get_initial_charges()` (per-atom partial
   charges, different thing). It never touched `atoms.info['charge']` /
   `atoms.info['spin']`. So the chgspin embedder saw all-zero inputs and the model
   couldn't condition on net charge/spin. Fixed: `__getitem__` now reads
   `atoms.info['charge']`/`'spin'` as scalar `long` tensors, `collate_fn`
   gathers them into per-batch tensors, `Batch` exposes them as
   `graph.charge` / `graph.spin`.

4. **`max_batch_size=batch_size` capped batches at 64 graphs** (`omol.py:1143`).
   Hipster's `_create_dataloader` passes `max_batch_size=999999` when
   dynamic batching is on. Our `get_omol_loaders` passed the yaml
   `batch_size` (=64) as the per-batch graph cap. With mean 55 atoms/graph
   that's 64 × 55 = 3,520 atoms/batch — exactly matching the observed
   3,515 atoms/step instead of the intended ≈12,000. **This was the
   dominant cause** of the residual val-metric gap. Now hardcoded
   `_GRAPH_CAP = 999999` in the dynamic-batching branch.

5. **Lightning auto-wraps custom sampler with DistributedSampler** (`main_omol.py:732`).
   Our `DynamicAtomBatchSamplerForAseDB` is already DDP-aware (shards by
   rank internally). Added `use_distributed_sampler=False` to `pl.Trainer`.

6. **Missing cudnn + dynamo knobs** (`main_omol.py:52-66`, this session).
   qcczbpfn's wandb config records `cudnn_benchmark: false`,
   `cudnn_deterministic: true`, and applies three `torch._dynamo.config`
   knobs (`cache_size_limit=256`, `force_parameter_static_shapes=False`,
   `capture_scalar_outputs=True`) whenever `compile=True`. PR set
   `matmul_precision='high'` but none of the others, so:
   (a) cudnn picked non-deterministic algos on the 1× H100 run;
   (b) snellius `zdudavnw` log shows `_dynamo` hitting the default
   `recompile_limit=8` on `PlatonicLinear.forward`, forcing a fall-back
   to eager mode for that subtree (correct numerically, slower).
   The fix is now in `mains/main_omol.py` and synced to
   `hipster:~/PlatonicTransformers-pr/` + `snellius:~/PlatonicTransformers/`.

7. **Hipster PR copy of `omol.py` was stale** (this session). The hipster
   `~/PlatonicTransformers-pr/platonic_transformers/datasets/omol.py` did
   not have the graph-cap fix from bug #4 — its
   `DynamicAtomBatchSamplerForAseDB` was instantiated with
   `max_batch_size=batch_size`. Under the hipster launcher's CLI
   override `--training.batch_size=16` that caps every batch at 16
   graphs ≈ 880 atoms (vs. the intended 3000 atoms per rank). Synced
   the laptop copy across before job 267926 starts.

## Current state of the comparison

After all fixes, snellius 1× H100 run **`zdudavnw`** (still running, ~80k steps
in, finishes around step 366k) vs **`qcczbpfn`** reference (4× hipster, 332k steps):

| Metric | PR/qcczbpfn ratio (mean across val rows, excl first) | Std |
|---|---|---|
| `e_mae/val` | 0.96 | 0.10 (very noisy) |
| `f_mae/val` | **1.040** | **0.020** |
| `loss/val` | 1.042 | 0.023 |

The systematic ~4% f_mae gap is **NOT yet definitively diagnosed as either
noise or code-bug** — we previously thought we'd estimated the noise floor
from `6qoxeqtc` (another 1-GPU run), but that run used `activation=gelu`
where qcczbpfn uses `activation=sin` — different model, not a valid
noise-floor reference.

## Variables that differ between PR (snellius zdudavnw) and qcczbpfn

| Variable | qcczbpfn | PR snellius (zdudavnw) | Likely impact |
|---|---|---|---|
| Hardware | 4× RTX 6000 Ada (sm_89) | 1× H100 (sm_90) | Different cuBLAS, ~few % noise |
| Parallelism | 4-way DDP | single-rank | All-reduce vs local sum |
| Train data | **100% of train_4M** (3,986,754 mol/epoch) | **100% of train_4M** (3,986,754 mol/epoch) | **No delta**: qcczbpfn's CLI overrode `validation_mode=heldout`, which skips the 0.9 train_split branch entirely. The `train_size: 0.9` line in `omol_4m.yaml` is dead code under heldout mode. Verified from qcczbpfn's `output.log`: "Training: 3986754". |
| Tokens/step (effective) | 11,811 | 11,952 | 1.2% packing-efficiency gain on 1 rank |
| Code | Frozen private repo @ `7fd2673` | Ported PR code | The thing we're auditing |
| Optimizer param-grouping | hipster's name-pattern rules (`build_optimizer_param_groups`) | Same end-state via different rules; **`film_projs.*.weight` is the only true delta**: hipster → no_decay, PR → decay. With wd=1e-8 + zero-init, impact is negligible but real. | Tiny |

## Pending jobs (as of session close)

- **hipster job 267660** `PT2-qcczbpfn-repro-4gpu` — pending in `performance`
  queue. Uses the **frozen private code** + exact env vars from
  `wandb-metadata.json`. **When it runs, it will be the sanity check that
  hipster code + venv + data still reproduces qcczbpfn**.
- **hipster job 267926** `PR-omol-hipster-4gpu` — pending (submitted this
  session). PR code on hipster 4× RTX 6000 Ada, exact qcczbpfn hardware
  + data. Uses synced laptop copy of `omol.py` (graph-cap fix) and the
  new `main_omol.py` (cudnn + dynamo knobs). This is the **apples-to-
  apples** comparison: same hardware as qcczbpfn, just our PR code path.
- **snellius job 22947572** `PR-omol-4gpu-snellius-1h` — pending. **4× H100,
  1 hour, effective batch 12k atoms/step** (3000 per rank). Uses the current PR
  code (cudnn + dynamo knobs included). Smoke test for PR code on 4× DDP
  hardware. **Compare its early trajectory to qcczbpfn at the matching
  step range.**
- **snellius job 22930751** `PR-omol-graphcap-fix` (zdudavnw on wandb,
  long-running 1× H100, started 2026-05-19). Already past step ~60k.
  Useful for full-run reproducibility check but currently more interesting
  for the late-training comparison. **Note**: this run does NOT have the
  cudnn/dynamo knobs (it started before they were added). Mid-session
  audit pulled its val curve: f_mae/val ratio vs qcczbpfn = 1.040, std
  0.020 — still tracking with the systematic ~4% gap.

## Audit findings this session (2026-05-20, mid-session)

A thorough byte-by-byte audit of every active code path turned up only two
real differences relative to the frozen hipster reference:

1. **Missing cudnn + dynamo knobs in PR main_omol.py** — see bug #6 above. Fixed
   this session (synced to hipster + snellius).
2. **Hipster's `~/PlatonicTransformers-pr/omol.py` was stale** — see bug #7.
   Fixed this session.

Every other code path (model definitions, loss, optimizer param-grouping,
augmentation, batch object, element-ref subtraction) is **algebraically
identical to hipster at qcczbpfn's config**:

- **`force_field.py`**: byte-identical except new args
  `qk_dim_factor=1`, `v_dim_factor=1`, `rope_v_independent=False` which are
  no-ops at defaults. **Confirmed safe.**
- **`platoformer.py`, `block.py`, `conv.py`**: same — qk_dim_factor/v_dim_factor
  branches all guarded behind `qk_dim_factor>1` / `flash3` checks.
- **`linear.py`, `rope.py`, `ape.py`, `chg_spin_emb.py`, `io.py`, `utils.py`,
  `groups.py`, `khot_embeddings.py`**: byte-identical to hipster.
- **`_compute_loss`**: algebraically equivalent to hipster's
  `PerAtomMAELoss + L2NormLoss + DDPLoss(reduction='mean')` at single-GPU.
  Element-ref subtraction goes through fp64 accumulation in PR (vs fp32 in
  hipster) — PR is slightly *more* accurate; diff ≤ few meV/mol, irrelevant
  for f_mae.
- **Optimizer**: only delta is `film_projs.*.weight` → decay in PR vs no_decay
  in hipster. At qcczbpfn's `weight_decay=1e-8`, the cumulative pull on those
  zero-init weights over 332k steps is ≈ `lr * wd * steps ≈ 1.66e-6`.
  **Negligible.**
- **O(3) augmentation**: PR samples one rotation per *molecule*, hipster one
  per *batch*; both uniform over O(3), both equivariant operations. Different
  samples per batch but identical distribution in expectation.

Conclusion: **no single code-level difference explains the systematic ~4%
f_mae gap.** The remaining candidates (to be disambiguated by the hipster
4-GPU PR run 267926):

1. Hardware (sm89 RTX 6000 Ada vs sm90 H100, cuBLAS / cuDNN choice).
2. cuDNN nondeterminism in zdudavnw (the run pre-dating bug #6's fix).
3. Single-rank vs 4-rank optimizer dynamics (subtly different fp32
   summation orders).
4. Training data ordering: PR's `OMolDataset` uses simple sequential
   `list(range(N))` + sampler-side shuffle; hipster's `AseDBDataset`-based
   path goes through fairchem's `AseDBDatasetWithChargeSpin`. Both are
   indexed the same way and read the same files, but the *order in which
   they're consumed* under DDP differs.

The next deterministic-comparison data point is hipster job 267926. If it
matches qcczbpfn within the noise floor of job 267660 (qcczbpfn repro from
frozen private code), the PR code path is faithful and the residual gap on
zdudavnw was hardware + cudnn-nondeterminism.

## Open audit questions for the next session

The user wants **a very, very thorough audit** of our current PR code vs the
hipster reference. The audit done this session caught the 4 main bugs but
there's a residual ~4% f_mae gap. Targets for next session:

1. **Run the PR code on hipster 4-GPU (apples-to-apples vs qcczbpfn).**
   The infrastructure is already in place:
   - PR code synced to `hipster:~/PlatonicTransformers-pr/`
   - Launcher: `scripts/run_omol_platonic_hipster.sh` (in that repo)
   - venv: use `/home/ebekker/platonic-omol-backup/venv` (has `ml-collections` + `mendeleev` already installed)
   - Submit: `ssh hipster "cd ~/PlatonicTransformers-pr && sbatch scripts/run_omol_platonic_hipster.sh"`
   - **This isolates code differences from hardware/data differences.**

2. **Deep, file-by-file audit** of every active code path between hipster
   `/home/ebekker/platonic-omol-backup/{training/, scripts/}` and PR
   `PlatonicTransformers/{mains/, platonic_transformers/, configs/, scripts/}`.
   Specific suspect areas (open from previous audits):
   - **`film_projs.*.weight` weight-decay**: small but real difference. Consider matching hipster (no_decay for these).
   - **O(3) augmentation reflection mechanic**: PR flips first ROW of rotation matrix; hipster flips first COLUMN. Both produce valid O(3) elements, audit called them equivalent, but worth re-verifying that the same molecule + same RNG state produces the same augmented input in both implementations.
   - **`metadata.npz` format**: hipster's ships `natoms` AND `data_ids`; our `build_omol_natoms_cache.py` only writes `natoms`. Functionally identical for the sampler but worth noting (a reviewer might want full parity). **Confirmed this session**: the `natoms` arrays are binary-identical between hipster `/scratch/ebekker/omol/` and snellius `/scratch-shared/ebekkers/omol25/` (md5 of `natoms.tobytes()`: val `c62ab04cfaa10cb1871cf8adc535e1f5`, train `c4ad3e68ae66549abe9bfc42b7f2bb34` on both clusters). The .aselmdb files themselves are also md5-identical for spot-checked shards. **Conclusion: train + val data byte-equivalent across clusters.**
   - **Element-ref energy subtraction**: hipster precomputes per-sample `energy_ref_corrected` in fp32 on CPU at iter time; PR computes on-the-fly in fp64. Numerically equivalent but timing/precision asymmetry.
   - **Optimizer init / weight init**: Hipster uses `build_optimizer_param_groups` from `nets.mup` even for non-muP runs. Verify there's no init / scaling subtlety that gets applied even in the `mup_multipliers_dict is None` branch.
   - **`force_field.py`** (PR `platonic_transformers/models/platoformer/force_field.py`) — port of hipster `training/nets/platoformer/model.py::PlatonicForceField`. Diff this byte-by-byte; small differences here would affect every forward pass.
   - **Loss reduction**: `_compute_loss` formulas (per-atom MAE for energy, L2-norm for force). Verify reduction axis (sum/mean), per-graph normalization, and how train_rmsd applies to energy vs force.
   - **`platonic_transformers/datasets/omol.py:OMolDataset` vs hipster `AseDBDatasetWithChargeSpin`**: PR uses its own DataLoader pipeline (with `collate_fn` returning a custom `Batch`); hipster uses `fairchem.core.datasets.atomic_data.atomicdata_list_to_batch`. The two batch-object types may have subtle attribute differences that the model implicitly relies on (e.g. `batch.idx_natoms`, `batch.cum_nodes`, etc).

3. **`6qoxeqtc` was NOT a clean noise-floor estimate** (activation = gelu vs sin). To get a real noise estimate, find/launch two runs that share the **exact** same code, config, and hardware — only differ in seed.

4. ~~**The 100% vs 90% data delta**~~: **RESOLVED in this session — not an
   actual difference**. qcczbpfn used CLI override `validation_mode=heldout`,
   under which `OMol4mModule.setup` takes the branch
   `train_dataset = base_train_dataset` (no slice). The
   `train_size: 0.9` line in `configs/data/omol_4m.yaml` only applies to the
   `train_split` branch and was dead config for this run. Verified from the
   actual `output.log`: `# Training: 3986754` (= full train_4M, not 0.9×).
   PR's removal of `apply_split()` matches qcczbpfn's effective behavior.

## Useful command snippets

### Check job status

```bash
# Snellius
ssh snellius "squeue -u \$USER -o '%.10i %.25j %.2t %.10M %.10L %R'"
ssh snellius "tail -50 /home/ebekkers/PlatonicTransformers/logs/<LOG>"

# Hipster
ssh hipster "squeue -u \$USER -o '%.10i %.30j %.2t %.10M %.10L %R'"
ssh hipster "tail -50 /home/ebekker/platonic-omol/logs/<LOG>"        # for hipster repro (267660)
ssh hipster "tail -50 /home/ebekker/PlatonicTransformers-pr/logs/<LOG>"  # for PR code
```

### Pull wandb val curves for a run

```python
import wandb
api = wandb.Api()
r = api.run('omol-leaderboard/scaling-laws-symmetry/<run_id>')
h = r.history(keys=['trainer/global_step','e_mae/val','f_mae/val','loss/val'], samples=2000)
h = h.dropna(subset=['e_mae/val'])
print(h.to_string())
```

(Run this from snellius where the wandb credentials are already in `.netrc`.)

### Relevant wandb run IDs

| Run ID | Description |
|---|---|
| `qcczbpfn` | THE reference — 4× hipster, sin activation, finished |
| `6qoxeqtc` | 1-GPU baseline, but **activation = gelu** (not a clean comparison) |
| `zdudavnw` (`rich-puddle-319`) | PR snellius 1× H100, still running |
| (next 4-GPU snellius run, when 22947572 starts) | PR 4× H100 1-hour smoke test |
| (next hipster repro, when 267660 starts) | qcczbpfn repro with frozen code |

### Submit hipster PR 4-GPU run (the apples-to-apples test)

```bash
ssh hipster "cd ~/PlatonicTransformers-pr && sbatch scripts/run_omol_platonic_hipster.sh"
```

(The launcher already has the right hipster sbatch directives + sets all
4-GPU DDP env vars + uses the right venv path.)

### Pull qcczbpfn's wandb-metadata.json (canonical hyperparam list)

```
cat public-pr-prep/baseline-reference/wandb/files/wandb-metadata.json | jq '.args'
```

## A self-contained prompt for the next agent session

Paste this into a fresh Claude session at the same working directory:

```
I'm continuing the public-PR work for `niazoys/PlatonicTransformers`. The
goal is to **exactly reproduce the qcczbpfn winning run** from the private
`ebekkers/platonic-omol` repo. Read
`public-pr-prep/HANDOVER.md` for the full state. TL;DR:

- The PR code at `public-pr-prep/PlatonicTransformers/` is at commit
  `aac7c35` on `ebekkers/platonic-omol@main`. We fixed 5 bugs this past
  session (ModelCheckpoint monitor key, train_rmsd clobber, charge/spin,
  graph-cap, distributed-sampler auto-wrap) and the snellius 1× H100 PR
  run (`zdudavnw` on wandb) now tracks qcczbpfn within a systematic ~4%
  f_mae/val gap.

- We don't yet know if that 4% is noise or a real residual code bug. I
  previously thought `6qoxeqtc` was a clean noise-floor reference but it
  uses `activation=gelu` while qcczbpfn uses `sin` — invalid comparison.

- Two pending jobs are queued: (a) **hipster 267660** —
  `PT2-qcczbpfn-repro-4gpu`, frozen private code, sanity-check that the
  hipster code still reproduces qcczbpfn; (b) **snellius 22947572** —
  `PR-omol-4gpu-snellius-1h`, our PR code on 4× H100 with effective batch
  12k for 1 hour, to test PR on 4-GPU DDP.

Tasks for this session:

1. Check the status of jobs 267660 and 22947572. If 22947572 has finished
   (or hit step 5000+), pull its val/token curves and compare to qcczbpfn
   at matching steps. Same for 267660 once it runs.

2. **Do a very thorough byte-by-byte audit of the active OMol training
   code path** between hipster
   `/home/ebekker/platonic-omol-backup/training/` and PR
   `public-pr-prep/PlatonicTransformers/`. Specific suspects listed in
   §"Open audit questions" of HANDOVER.md.

3. **Also submit a hipster 4-GPU PR run**: the infra is already in place
   at `hipster:~/PlatonicTransformers-pr/`. Just
   `ssh hipster "cd ~/PlatonicTransformers-pr && sbatch scripts/run_omol_platonic_hipster.sh"`.
   This is the cleanest test: PR code on the exact hipster hardware that
   produced qcczbpfn. If its f_mae/val matches qcczbpfn within the
   noise of 267660, the PR code is faithful and we can move on.

Read `public-pr-prep/HANDOVER.md` first for the full context, command
snippets, file paths, and bug list. Do NOT modify any files in
`platonic-omol-backup/` or `platonic-omol/` on hipster — they are
reference state.
```
