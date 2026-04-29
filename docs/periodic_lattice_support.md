# Periodic Lattice Support

Platonic RoPE normally applies Cartesian positions to queries and keys. For
non-periodic systems this gives relative-displacement attention scores. For
crystals, the phase should be unchanged by integer lattice translations.

This branch adds a lattice-aware attention path:

- `PlatonicTransformer.forward(..., lattice=cell, pbc=pbc)` threads cell
  matrices and periodic flags through every Platonic block.
- The default `lattice_rope_mode="reciprocal"` uses integer lattice harmonics
  in fractional coordinates. For row-vector cells `r = frac @ cell`, pairwise
  Cartesian displacements are converted back to fractional displacements, each
  harmonic phase `frac dot n` is wrapped modulo one, and the final RoPE angle
  is `2*pi*(frac dot n mod 1)`. Integer cell translations therefore produce
  exactly the same rotation up to floating-point roundoff.
- `lattice_rope_mode="minimum_image"` remains available for local
  nearest-image attention and partial periodic boundaries. It wraps
  `pos_j - pos_i` through fractional coordinates before applying the usual
  Cartesian RoPE frequencies.
- Softmax attention computes lattice RoPE pairwise from `pos_j - pos_i`,
  preserving the Platonic group action while making predictions invariant to
  integer unit-cell shifts. Reciprocal mode requires all `pbc` dimensions to be
  true.
- The factorized linear attention path raises a `ValueError` when `lattice` is
  supplied. That path aggregates a global key/value kernel and cannot exactly
  represent pair-specific lattice phases.

The smoke dataset and trainer in `platonic_transformers.datasets.mp20` and
`mains/main_mp20_smoke.py` provide a small CIF-backed periodic regression run.
The default CIF parser reads listed atom sites without symmetry expansion so
large local MP-style CIF directories can be tested quickly; pass
`parser="pymatgen"` or `--parser pymatgen` when full pymatgen parsing is
needed.

For full MP20 benchmark splits stored as CSV files with CIF strings and target
columns, use `MP20CSVRegressionDataset` or `mains/main_mp20_regr.py`. The
trainer defaults to the repo's full-size PlatonicTransformer architecture
(`hidden_dim=1152`, `num_layers=14`, `num_heads=72`) and reports periodic
integer-cell shift checks before and after training.
