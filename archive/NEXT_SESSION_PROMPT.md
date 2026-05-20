# Opening prompt for next session — Platonic OMol25 (IVI flash-attn runs)

Copy-paste this to start a fresh session:

---

You are helping with the **Platonic OMol25 leaderboard** project, this session focused on **getting useful runs going on IVI now that flash-attn is finally built** for the IVI GPUs (RTX 8000 / RTX A6000, sm_75/sm_86/sm_89). The Snellius runs from the previous session are running unattended on 24h walltimes — only check them if I ask.

**Important:** If working from the Nomi workspace, the project repo is nested at `workspaces/platonic-omol/platonic-omol/`. All paths below are relative to the inner repo root.

## Read first

1. `../README.md` — workspace README, top section "2026-05-16" has the current state.
2. `MEMORY` entries: `reference_snellius.md`, `feedback_commit_workspace_readmes.md`, `project_scaling_laws_symmetry.md`.
3. This file.

## Where things stand (2026-05-16, end of previous session)

### IVI: flash-attn finally builds for sm_86/89

After two failed attempts (wrong archs; MAX_JOBS=2 too slow), `scripts/install_flash_attn_ivi_v2.sh` patches `setup.py` to add sm_86 and sm_89 `cuda_archs()` branches alongside sm_80, then builds with MAX_JOBS=8 on a 12h walltime allocation. Build 168484 completed in 4h32m. `flash_attn 2.7.4.post1` is now installed in `/home/ebekker/platonic-omol/venv/` and imports cleanly.

This is a meaningful unlock: every prior IVI run used the scatter backend (2–2.5× slower than flash at matched precision on hipster's Ada GPUs, per the 2026-05-08 hipster benchmarks).

### IVI repo state

- `ssh ivi_cluster` lands as user `ebekker` at `~/`. Repo at `/home/ebekker/platonic-omol/`, branch `main`, HEAD `9f14fd4` ("pure-PyTorch radius_graph fallback…"). **This is behind laptop main (`80f26e5`).** First step on IVI is `git pull` — laptop has 3 new commits (qk_norm/swiglu, use_key, norm_type plumbing) that are not yet on IVI but matter only if you want to run the qknorm variants there.
- Modified-uncommitted on IVI: `scripts/install_flash_attn_ivi.sh` (older version of the install script). Safe to leave — `_v2.sh` is the one that worked.
- Venv: `/home/ebekker/platonic-omol/venv/` (Python 3.12). Activate via `source /home/ebekker/platonic-omol/venv/bin/activate`.

### Pre-existing IVI launchers (in `scripts/`)

- `run_pt2_upstream_long_sig4_dyn_ivi.sh` — older, scatter backend
- `run_pt2_upstream_long_sig4_ema_dyn_ivi.sh` — older + EMA
- `run_pt2_upstream_long_sig4_wd_bs_ivi.sh` — older wd/bs sweep
- `run_pt2_ivi_geodude_lsablation.sh` — recent ls=null ablation set, scatter
- `smoke_ivi.sh`, `smoke_ivi_all6000.sh`, `smoke_ivi_radius_lg.sh` — short smoke tests

None of these currently set `ATTENTION_BACKEND=flash`. **There is no "production" flash-enabled IVI launcher yet** — this is the first concrete task.

### Pending / waiting items

- **Task #9 (still pending): "IVI: submit LG long run after flash-attn builds."** The local-global launcher draft at `/tmp/run_pt2_ivi_all6000_lg.sh` (on laptop) still needs `MAX_ATOMS` bumped (4000 → 8000+) and to be uploaded to IVI. Local-global uses scatter for local sub-blocks regardless, so flash only buys the global half; still worth it.
- **Snellius (running unattended, do NOT touch unless asked):** 6 jobs on 24h walltime, all wd=1e-8 + LS=null + GeLU + chgspin-FiLM + rope_sigma=2.0 + ema=0.99, started 2026-05-16 ~07:30–08:30 UTC. W&B project `omol-leaderboard/scaling-laws-symmetry`.
  - `22783313` — GeLU baseline resume (from `tasrz3p1`'s last.ckpt, 18.6M, eW1/fW10, MAX_ATOMS=12000)
  - `22783309` — + qknorm + swiglu (23.5M, eW10/fW20, MAX_ATOMS=12000)
  - `22783311` — + qknorm + swiglu + use_key (23.5M, eW10/fW20, MAX_ATOMS=12000)
  - `22783342` — + qknorm + swiglu + **rmsnorm** (23.5M, eW10/fW20, MAX_ATOMS=12000) — this is the "best stack so far" candidate (W&B `lv7akpah`)
  - `22783550` — + qknorm + swiglu + rmsnorm, **num_layers=16** (46.1M, eW10/fW20, MAX_ATOMS=20000)
  - `22783551` — + qknorm + swiglu + rmsnorm, **hidden_dim=3840 / nhead=120** (94.1M, eW10/fW20, MAX_ATOMS=20000)
  - `22783623` — qknorm + swiglu + use_key, **bf16-mixed** + compile=on (26.0M, eW10/fW20, MAX_ATOMS=12000) — twin of `yxh5y61s`/22783311 to revisit the 2026-05-08 bf16 verdict now that qknorm is in the stack.
  - The L=16 and h=3840 runs are scale-up probes on top of `lv7akpah`. Both at MAX_ATOMS=20000 (Erik's call — H100 fits it at these sizes). With 24h walltime they will get fewer steps (16-layer ≈ 2× slower per step, 3840-hidden ≈ 4× slower) so a resume may be needed to fully complete 20 epochs — flag this if asked.

## Concrete things to do this session

Pick whichever makes the most sense given the day; ask me first if unsure.

### Option A — Get the LG long run on IVI launched (closes task #9)

1. Pull `/tmp/run_pt2_ivi_all6000_lg.sh` from laptop, bump `MAX_ATOMS` to 8000 (or 10000), copy to `ivi_cluster:~/platonic-omol/scripts/`.
2. Decide: flash-on-global yes/no. `local_global=true` keeps scatter for local sub-blocks, but global can switch to `attention_backend=flash`. The launcher needs `ATTENTION_BACKEND=flash` env var threaded through.
3. Submit on `all6000` partition. Watch for `flash_attn` import OK + Hydra parse OK in the first 60s of log.
4. Set a long watcher (use the fresh-SSH-per-poll pattern; IVI auto-kills idle SSH).

### Option B — Quick flash-vs-scatter A/B on IVI

To validate flash is actually faster on IVI GPUs (we have hipster Ada numbers but not IVI A6000 numbers): submit two short (1500-step) twin jobs, same recipe, only `ATTENTION_BACKEND` differing. Pulls W&B steady-state ms/step and atoms/sec. If flash wins by ≥1.5× on A6000, all future IVI launchers should default to flash.

### Option C — Promote qknorm/swiglu/rmsnorm wins from Snellius to an IVI long run

Wait until Snellius runs are 8–10h in (so they pass step ~5000 and we can pick a winner), then port the winning recipe to a fresh IVI launcher with flash. **This is wait-state today** — flag and check in a few hours rather than running.

## Don't touch

- **Snellius runs** (`22783313`, `22783309`, `22783311`, `22783342`) — they're the experiment, leave them alone.
- The `upstream-port-pt2` branch on IVI (ahead 21 of origin). Old work; not relevant to this session.

## How to check status quickly

```bash
# IVI queue
ssh ivi_cluster "squeue -u ebekker"

# Snellius queue (only if asked)
ssh snellius 'squeue -u ebekkers --format="%.10i %.42j %.10T %.10M %.13l"'

# Flash-attn import check on IVI
ssh ivi_cluster "source /home/ebekker/platonic-omol/venv/bin/activate && python -c 'import flash_attn; print(flash_attn.__version__)'"
```

**Ask me what we're working on today before starting any task.**
