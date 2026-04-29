# Periodic Lattice Support

Platonic RoPE normally applies Cartesian positions to queries and keys. For
non-periodic systems this gives relative-displacement attention scores. For
crystals, the phase should be unchanged by integer lattice translations.

This branch adds a lattice-aware attention path:

- `PlatonicTransformer.forward(..., lattice=cell, pbc=pbc)` threads cell
  matrices and periodic flags through every Platonic block.
- The default `lattice_rope_mode="reciprocal"` uses integer reciprocal-lattice
  harmonics. For row-vector cells `r = frac @ cell`, each mode `n` uses a
  Cartesian reciprocal vector `g` satisfying `cell @ g = 2*pi*n`. Therefore
  `r dot g = 2*pi*(frac dot n)`, and any integer cell translation changes the
  phase by an exact multiple of `2*pi`.
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
