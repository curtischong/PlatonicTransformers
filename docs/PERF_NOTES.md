# Performance notes: PlatoFormer + bf16 + torch.compile + dynamic batching

This document records what we learned getting bf16-mixed precision to actually
deliver a speedup on H100 for PlatoFormer training with dynamic-atom batching.
The naive Lightning incantation (`+trainer.precision=bf16-mixed`) made training
**3× slower than fp32**, not faster. The fix touched four places:
`torch._dynamo.config`, the `torch.compile` call site, the LightningModule's
per-step Python overhead, and the dataloader. Each of these lessons is
codebase-agnostic and worth porting to the public `platonic-transformers`
repository.

The starting symptom is sharp and easy to recognize, so the first section
describes how to diagnose; the rest explain why and what to fix.

---

## Diagnosing: the W&B + `TORCH_LOGS` recipe

The single most valuable tool here is W&B's per-process system metrics combined
with PyTorch's dynamo logging. Two signals tell you almost everything:

```bash
# Add to the launch script while diagnosing.
export TORCH_LOGS=recompiles,graph_breaks
```

Then look at:

1. **`system.gpu.0.gpu`** in W&B's "System" panel. A healthy training run on
   H100 sits at 90–100% GPU util. A run reading `0–10%` is CPU-bound, not
   compute-bound. (Power confirms: full util = ~600–700W, idle wait = 150–250W.)
2. The slurm log will print every dynamo recompile and graph break. Count
   them. If a single function recompiles more than ~8 times, dynamo has
   exceeded its default `cache_size_limit` and is silently falling back to
   eager — i.e. your `torch.compile` is decoration on the books, not in fact.

Concrete pattern from our broken bf16 run (job `22569081`, 21 minutes of
training):

| Recompile cause | Count | Source |
|---|---|---|
| `GLOBAL_STATE changed: grad_mode` | 80 | Lightning toggling no_grad/inference_mode |
| `tensor 'x' size mismatch at index 0` | 25 | dyn-batching producing new total-atom counts |
| `self._parameters['bias']` shape mismatch | 25 | shared `forward` across `PlatonicLinear` instances with different bias sizes |
| `tensor 'x' dtype mismatch (BFloat16 ↔ Float)` | 10 | autocast specializing one way, then the other |

Function-level: `PlatonicLinear.forward` hit **11 recompiles** — past dynamo's
default `cache_size_limit=8` — and was running in eager for the rest of the run.

---

## What "precision" actually is — three orthogonal knobs

These get conflated all the time. They are independent and compose:

### 1. `trainer.precision` — Lightning's wrapping of the forward pass

| Value | Meaning | When to use |
|---|---|---|
| `32-true` | pure fp32 (default) | reference, debugging |
| `bf16-mixed` | params fp32, autocast → activations bf16, loss fp32, optimizer fp32 | **production on Hopper/Ampere** |
| `bf16-true` | params bf16 module-wide, no autocast, optimizer bf16 | memory-constrained; risks optimizer drift |
| `16-mixed` | fp16 + GradScaler | don't use on Hopper — bf16-mixed strictly better |

### 2. `torch.set_float32_matmul_precision(...)` — how `aten::mm` handles fp32 inputs

| Value | Meaning |
|---|---|
| `highest` | true fp32 matmul (~67 TFLOPS on H100) |
| `high` | TF32: fp32 inputs/outputs, bf16-mantissa multiply on tensor cores (~989 TFLOPS) |
| `medium` | bf16 matmul for fp32 inputs (lower precision than `high`) |

This is **independent of `trainer.precision`**. Even pure-fp32 training benefits
from `matmul=high` — TF32 ≈ fp32 accuracy in practice with a ~15× speedup on H100.

### 3. cuDNN backend — `cudnn.benchmark` and `cudnn.deterministic`

`benchmark=True` autotunes the best kernel for each unique input shape.
**With dynamic batching this is questionable** — shapes vary every batch, so
the autotune cache thrashes. Most attention in PlatoFormer flows through
`flash_attn_varlen_func`, not cudnn, so it doesn't help much there either.
We leave it on with the bf16 path and it didn't hurt, but consider it tunable.

In Hydra, expose these as three independent keys (not bundled), with two
named presets (`bf16_h100`, `fp32_baseline`). See `configs/precision/` and
`configs/train_omol.yaml`.

---

## The dynamo recompile-thrash trap

`torch.compile` traces a graph and guards on the inputs. When a guard fails,
dynamo recompiles. With four sources of guard variance compounding, the cache
fills up fast:

1. **Dynamic batching** produces a fresh total-atom count nearly every batch.
   Default behavior: dynamo specializes on the first shape, then on shape
   mismatch retraces with `dynamic=True` (per `automatic_dynamic_shapes`).
   But automatic mode tries specialization first — slow path.

2. **Shared `forward` across instances with different parameter shapes**.
   `PlatonicLinear.forward` is the same Python function for ~65 instances
   in a 12-layer PlatoFormer (q/v/out_proj per layer × 12 + 2 FFN linears
   × 12 + embedder + readouts). Each instance has different bias sizes
   (144, 576, 1, etc.). Dynamo guards on `self._parameters['bias'].size(0)`
   so each unique bias shape is a recompile.

3. **Lightning toggles `grad_mode`** between train, sanity-val, val,
   `on_train_epoch_start`, and EMA-style hooks. Each toggle counts as
   `GLOBAL_STATE changed` and invalidates all guards.

4. **Autocast dtype non-determinism**. Under `bf16-mixed`, the same callsite
   sometimes receives fp32 (before the autocast cast happens) and sometimes
   bf16 (after). Dynamo specializes once, then recompiles when the dtype
   flips, then flips back, etc.

Default `cache_size_limit=8` is far too small for this combination.

### The four required knobs

```python
import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 256
dynamo.config.force_parameter_static_shapes = False
dynamo.config.capture_scalar_outputs = True

torch.compile(model, dynamic=True)  # not None (auto), not False — explicit
```

What each does:

- **`cache_size_limit = 256`** — enough headroom for the multiplicative cross
  product of (shape buckets × bias sizes × dtypes × grad_mode states).
- **`force_parameter_static_shapes = False`** — stop guarding on parameter
  shapes; lets one compiled function service all instances. PyTorch suggests
  this in the recompile log itself when triggered.
- **`capture_scalar_outputs = True`** — absorbs `Tensor.item()` calls into
  the compiled graph. Critical for FA-varlen, which calls `int(batch.max().item())`
  and `int(counts.max().item())` to compute `cu_seqlens` / `max_seqlen`. Without
  this, every layer's attention is a graph break.
- **`compile(model, dynamic=True)`** — tells dynamo to handle dynamic shapes
  natively from the start, avoiding the "specialize-once-then-make-dynamic"
  warmup and a class of associated recompiles.

In our codebase these are auto-applied when `force_field_module.compile=true`
(see `train_omol.py`), since they're intrinsic to making compile actually work
on this model + dyn-batching, not separate optional tuning.

---

## Autocast pitfall: RoPE under bf16

This was the second autocast trap, and it doesn't hurt throughput at all — only
accuracy. It cost us a 4-hour bf16 production run that converged to a strictly
worse force-MAE floor than fp32 (~135 meV/Å vs ~37 meV/Å at matched steps).

### The setup

`PlatonicRoPE.forward` (in `rope.py`) computes positional encoding angles as

```python
freqs_rotated = torch.einsum('gde, hfe -> ghfd', self.group_elements, self.freqs)
angles = torch.einsum('...d, ghfd -> ...ghf', pos, freqs_rotated)
cos_angles = torch.cos(angles)
sin_angles = torch.sin(angles)
```

Under `bf16-mixed`, both einsums sit in autocast's "lower precision" list, so
`freqs_rotated`, `angles`, and the trig outputs all run in bf16.

### Why it's catastrophic for *this* model

For frequency stddev `σ = rope_sigma` and atomic position `p`, the angle
`θ = p · (R_g · f)` is a 1D Gaussian:

$$\theta \sim \mathcal{N}(0,\; \sigma^2 \|p\|^2)$$

For `rope_sigma=4` and Å-scale positions (‖p‖ ≈ 5–15 Å on OMol25 medium
molecules), the stddev of θ is **20–60 radians**. The tail reaches over 100.

bf16's 8-bit mantissa gives spacing ~`|x| · 2⁻⁸` between representable values:

| `|θ|` | Δθ in bf16 | Δcos (worst case) |
|---|---|---|
| 1 rad | 0.004 | ~0.004 (fine) |
| 10 rad | 0.04 | ~4% |
| 40 rad | 0.16 | ~16% (catastrophic) |
| 100 rad | 0.4 | unbounded |

Multiply that error per layer (12) and across both q-rotation and v-rotation
(`rope_on_values=True`), and the positional encoding is scrambled. Forces
collapse to a much worse floor than fp32 because forces depend on the
directional/angular structure that RoPE provides.

### The signature in W&B

`f_mae/val` divergence between bf16 and fp32 *widens* during training rather
than narrowing, and bf16 plateaus far above the fp32 floor:

| step | fp32 f_mae (meV/Å) | bf16 f_mae (meV/Å) |
|---|---|---|
| 5,000 | 132.8 | 223.4 |
| 26,611 | 54.4 | 153.1 |
| 48,223 | 41.2 | 141.0 |
| 98,000 | (still descending) | 135.1 ← floor |

Throughput is fine — bf16 still ~5× faster per step. The pathology is purely
on the accuracy side.

### The fix

Wrap the einsums + trig in `autocast(enabled=False)`, force fp32 inputs, cast
back to the input dtype only for the rotation application:

```python
with torch.amp.autocast(device_type=x.device.type, enabled=False):
    freqs_rotated = torch.einsum(
        'gde, hfe -> ghfd', self.group_elements, self.freqs.float()
    )
    angles = torch.einsum('...d, ghfd -> ...ghf', pos.float(), freqs_rotated)
    cos_angles = torch.cos(angles).to(x.dtype)
    sin_angles = torch.sin(angles).to(x.dtype)
```

The cos/sin tensors are small (`B × num_atoms × G × H × num_pairs`); the fp32
cost is negligible. The rotation itself stays in bf16 / autocast.

### Why this isn't an "inherent bf16 limitation"

bf16 is fine for the multiplications and accumulations in attention, FFN,
LayerNorm, etc. — the model dynamic range there is not extreme. RoPE is the
exception because its inputs (angles in radians) can have very large magnitude
relative to the function's natural scale (one period = 2π). This is identical
to the issue LLM RoPE implementations hit when ported to bf16; the
corresponding LLM-RoPE codebases all wrap the trig in fp32. PlatoFormer
inherited the autocast vulnerability when we turned on `bf16-mixed` and didn't
follow that pattern.

### Generalizable lesson

If a layer mixes large-magnitude angles, exponentials, or any other function
whose precision degrades sharply with input magnitude, force fp32 inside it.
Other candidates worth auditing in any new codebase:

- LayerNorm internals (PyTorch's autocast policy already handles this).
- Loss-function reductions over many terms (sums of bf16 tensors lose precision
  beyond ~256 terms).
- Anything computing `softmax`, `log`, or `exp` on values with large dynamic
  range. FlashAttention does internal fp32 accumulation; pure-bf16 manual
  attention does not.

---

## Why caching `PlatonicLinear.get_weight()` doesn't help (a debugging dead end)

This trap cost us a turn of analysis, so it's worth documenting.

`PlatonicLinear` reconstructs its dense weight from a smaller structured
`self.kernel` parameter via gather + permute + reshape on every forward call:

```python
def forward(self, x):
    weight = self.get_weight()      # fresh tensor each call
    return F.linear(x, weight, None)
```

Initial hypothesis: under `bf16-mixed`, autocast must cast `weight` from fp32
to bf16 on every call. Autocast's cast cache is keyed on tensor storage pointer,
so a fresh tensor each call defeats the cache.

Reality: **`PlatonicLinear` is called once per instance per forward pass.**
Caching the materialized weight across calls within a forward doesn't reduce
cast count, because there's no repeat-call pattern within a single forward.
The cast happens once per instance per forward in either case. The cache-miss
math came out to ~0.5 ms of extra memory traffic per step, nowhere near the
1500 ms regression.

The actual cause of the regression was the dynamo recompile thrash above.
Lesson: **measure the suspected fix's contribution before implementing it.**
Storage-pointer cache misses sound expensive but are cheap if you only call
each function once anyway.

---

## Reclaiming GPU util in bf16: the per-step CPU overhead is now visible

In fp32 on H100, this model ran at ~95% GPU util because each fp32 matmul
took ~700 ms of compute and CPU overhead (logging, augmentation, optimizer
step) was hidden inside that window. Switching to bf16 cut compute to ~50 ms.
The same CPU overhead is now exposed and GPU util drops to ~26%.

Five places per training step were costing us:

### 1. Per-parameter grad-norm logging in `on_after_backward`

```python
def on_after_backward(self) -> None:
    if self.trainer.global_step % 100 == 0:
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().norm(2).item()  # CUDA→CPU sync
                self.log(f"grad_norm/{name}", grad_norm)
```

The 33.7M-parameter PlatoFormer has ~300–500 distinct parameter tensors. This
hook fires 500 `.item()` syncs back-to-back every 100 steps and creates 500
unique W&B keys (`grad_norm/...`). Single biggest non-compute event in the
training loop. **Just delete it** — it's debugging infrastructure that should
not ship in the production training loop. Use a profiler or a one-off analysis
hook when you actually need grad norms.

### 2. Per-step `torch.isfinite(loss).all()` NaN guard

```python
if not torch.isfinite(loss).all():
    raise ValueError("NaN detected in loss")
```

The `if not ...` Python branch forces a CUDA flush every step. Trivial
in fp32 (1 ms / 700 ms ≈ 0.1%); meaningful in bf16 (1 ms / 150 ms ≈ 1%).
Remove it; rely on `gradient_clip_val` plus the loss curve in W&B as the
NaN-detection mechanism.

### 3. Many-individual-`self.log` calls in the training step

Lightning's `self.log()` has nontrivial Python overhead per call (computes
metric, registers with the logger, dispatches sync_dist). With 7 metrics
times 2 (train + epoch logging) plus 3 token-tracking logs, we had ~10–14
`self.log` calls per step.

Fix: batch them into a single `self.log_dict({...})` call. Same semantics,
one Python dispatch instead of N.

### 4. CPU-side random rotation augmentation

```python
H = torch.randn(3, 3)            # CPU
Q, R = torch.linalg.qr(H)        # CPU
if torch.det(Q) < 0:             # branch on .item() implicitly
    Q[:, 0] *= -1
R = R.to(batch.pos.device)       # H2D transfer per step
```

Move it to the batch's device. Use `torch.where(torch.det(Q) < 0, -1.0, 1.0)`
instead of the Python branch to avoid the host sync.

### 5. Dataloader workers ≪ allocated CPUs

We had `--cpus-per-task=16` but `num_workers=8` and `prefetch_factor=2` —
8 cores idle and 16 batches in flight. With bf16 chewing through batches
every 150 ms (vs 700 ms in fp32), the prefetch queue drains faster than
workers can refill.

Fix: `num_workers=16, prefetch_factor=4`. The risk of saturating workers up
to allocated CPUs is low for I/O-bound LMDB readers — workers spend most of
their time blocked on disk, not competing for CPU. With `persistent_workers=True`
(already on), there's no per-epoch startup cost.

### What this looked like in numbers

| Run | Mean GPU util | ≥80% sample share | Power |
|---|---|---|---|
| Original fp32 baseline (jtvde113) | 95% | 97% | 623W |
| Broken bf16 (no fixes) | 6.7% | 5% | 154W |
| bf16 + dynamo knobs | 26% | 22% | 257W |
| bf16 + dynamo knobs + lean module + 16 workers | **41%** | **43%** | **312W** |

bf16 still has lower mean util than fp32 (no surprise — fp32 compute time is
the floor), but at ~150 ms/step it does ~5× more steps per wallclock. Net
training-progress-per-day is the right metric, not per-step GPU util.

---

## Hydra plumbing: precision as first-class config

Replace opaque environment variables with config keys. Three top-level keys:

```yaml
matmul_precision: highest    # → torch.set_float32_matmul_precision
cudnn_benchmark: false
cudnn_deterministic: true
```

Two preset files in `configs/precision/`:

```yaml
# bf16_h100.yaml
# @package _global_
trainer:
  precision: bf16-mixed
matmul_precision: high
cudnn_benchmark: true
cudnn_deterministic: false
force_field_module:
  compile: true
  compile_dynamic: true
  compile_mode: default
```

```yaml
# fp32_baseline.yaml
# @package _global_
trainer:
  precision: 32-true
matmul_precision: high       # keep TF32 — essentially free
cudnn_benchmark: false
cudnn_deterministic: true
force_field_module:
  compile: false
```

Selected at launch with `+precision=bf16_h100` or `+precision=fp32_baseline`.

The `# @package _global_` directive flattens the preset's keys to the global
config root, so a single line on the launch command sets values across the
`trainer`, `force_field_module`, and top-level blocks at once.

In `train_omol.py`, read from cfg and apply:

```python
torch.set_float32_matmul_precision(cfg.matmul_precision)
torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
torch.backends.cudnn.deterministic = bool(cfg.cudnn_deterministic)

if cfg.force_field_module.compile:
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 256
    dynamo.config.force_parameter_static_shapes = False
    dynamo.config.capture_scalar_outputs = True
```

In the LightningModule's `setup()`:

```python
if self.hparams.compile and stage == "fit":
    self.net = torch.compile(
        self.net,
        mode=self.hparams.compile_mode or "default",
        dynamic=self.hparams.compile_dynamic,
    )
```

---

## Porting checklist for the public `platonic-transformers` repo

For someone applying these lessons to a fresh codebase:

- [ ] Add `matmul_precision`, `cudnn_benchmark`, `cudnn_deterministic` as top-level
      Hydra (or argparse) config keys with safe defaults. Default to `highest /
      false / true` (Lightning + PyTorch defaults — reproducible).
- [ ] Add `compile`, `compile_mode`, `compile_dynamic` to the model config.
      Default `compile=false` so the slow-but-reproducible path is the default.
- [ ] Provide a `bf16_h100` preset (or whatever the production hardware is) that
      flips them all to fast settings in one line.
- [ ] In the training entry point, when `compile=true`, auto-apply the four
      dynamo knobs. Don't make them user-tunable — they're prerequisites, not
      preferences.
- [ ] In the LightningModule:
  - [ ] No per-parameter grad-norm logging in `on_after_backward`.
  - [ ] No `torch.isfinite(loss).all()` per step (rely on `gradient_clip_val`).
  - [ ] Use `self.log_dict({...})` for batched metric logging.
  - [ ] Run any per-step augmentation on the batch's device, with `torch.where`
        instead of Python branches.
- [ ] Set `num_workers = --cpus-per-task` (or close to it) and
      `prefetch_factor=4` for fast-step regimes. Verify `persistent_workers=True`.
- [ ] In `PlatonicConv.graph_flash_varlen_attention` (and equivalents), be aware
      that `int(batch.max().item())` and `int(counts.max().item())` are graph
      breaks. They are absorbed by `dynamo.config.capture_scalar_outputs=True`
      but only when compile is on.
- [ ] In `PlatonicLinear.forward`, leave `get_weight()` as-is. Caching the
      result is tempting but doesn't reduce cast traffic when each instance is
      called exactly once per forward.
- [ ] In `PlatonicRoPE.forward` (and any other code computing trig of large
      angles), wrap the einsums + sin/cos in `torch.amp.autocast(enabled=False)`
      and cast the trig outputs back to the input dtype before the rotation.
      Small fp32 cost, prevents the bf16 RoPE-precision pathology that floors
      force-MAE several × above fp32.

---

## Validation runs

Two long runs are scheduled in parallel under W&B group
`pt2-sig4-wd1e4-precision-comparison` to validate that bf16 doesn't sacrifice
leaderboard accuracy:

- `pt2-sig4-wd1e4-bf16` (job 22570079): `+precision=bf16_h100`, 5d
- `pt2-sig4-wd1e4-fp32` (job 22570080): `+precision=fp32_baseline`, 5d

Same recipe, same code, only the precision preset differs. Compare on:

- `loss/val` and `f_mae/val` vs `trainer/global_step` (per-step accuracy)
- `loss/val` and `f_mae/val` vs `_runtime` (per-wallclock accuracy — the metric
  that actually decides which preset to ship)
- Final full-validation `f_mae/test` from `test_omol.py` on each checkpoint
