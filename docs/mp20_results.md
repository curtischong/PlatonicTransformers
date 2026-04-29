# MP20 Periodic Regression Results

These runs use the local MP20 CSV benchmark splits at
`/data/adeesh/crystal-llm-v2/resources/benchmarks/mp_20`:

- Train: 27,136 structures
- Validation: 9,047 structures
- Test: 9,046 structures
- Target: `formation_energy_per_atom` in eV/atom

## Results

| Model | Training data | Val MAE | Test MAE | Runtime |
| --- | --- | ---: | ---: | ---: |
| Periodic PlatonicTransformer | MP20 train split only | 0.09854 | 0.09976 | ~31 min |
| ExtraTrees composition/lattice baseline | MP20 train split only | 0.10960 | 0.11051 | 28.4 sec |

The PlatonicTransformer result is from `checkpoints/mp20_regr_full/best.pt`
at epoch 19. It used the full-size repo configuration:
`hidden_dim=1152`, `num_layers=14`, `num_heads=72`, batch size 32, bf16,
and strict periodic fractional RoPE.

The ExtraTrees baseline is reproducible with:

```bash
python -u mains/main_mp20_sklearn_baseline.py \
  --csv-dir /data/adeesh/crystal-llm-v2/resources/benchmarks/mp_20 \
  --checkpoint-dir checkpoints/mp20_sklearn_baseline
```

Its metrics are written to `checkpoints/mp20_sklearn_baseline/metrics.json`.
It uses 249 MP20-only features: composition fractions, element-presence flags,
cell lengths/angles, volume features, atom count, and space-group number.

## Periodicity Checks

For the best PlatonicTransformer checkpoint:

- Test periodic integer-cell shift max diff, fp32: `3.576e-07`
- Test periodic integer-cell shift max diff, bf16: `1.270e-02`

The bf16 value reflects reduced-precision inference roundoff; the fp32 path
shows the strict periodic wrapping is working.

## Training Time

The original 30-epoch PlatonicTransformer run was launched before trainer-side
timing was added. From the command session and `last.pt` checkpoint timestamp,
it took approximately 31 minutes on one H100 80GB GPU, or about 62 seconds per
epoch. `mains/main_mp20_regr.py` now records exact per-epoch and total runtime
in `metrics.json` for future runs.

## External Context

Published Materials Project formation-energy models are usually trained on
larger MP snapshots rather than the MP20 split used here. For example, the
MEGNet documentation reports MP-2018.6.1 and MP-2019.4.1 formation-energy MAEs
of 0.028 and 0.026 eV/atom, respectively, on much larger MP datasets. Those
numbers are useful context but are not apples-to-apples with this MP20-only
experiment.

Source: https://materialsvirtuallab.github.io/megnet/
