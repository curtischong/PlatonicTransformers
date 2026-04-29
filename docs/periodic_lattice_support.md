# Periodic Lattice Support

Platonic RoPE normally applies absolute Cartesian positions to queries and
keys. For non-periodic systems this is equivalent to a relative displacement in
the attention score. Periodic crystals need a pair-specific relative
displacement because the nearest image of atom `j` can differ for each query
atom `i`.

This branch adds a lattice-aware attention path:

- `PlatonicTransformer.forward(..., lattice=cell, pbc=pbc)` threads cell
  matrices and periodic flags through every Platonic block.
- `minimum_image_displacement` converts Cartesian pair displacements to
  fractional coordinates, wraps periodic dimensions, and converts back to
  Cartesian vectors.
- Softmax attention computes RoPE from the minimum-image displacement
  `pos_j - pos_i` directly, preserving the Platonic group action while making
  predictions invariant to integer unit-cell shifts.
- The factorized linear attention path raises a `ValueError` when `lattice` is
  supplied. That path aggregates a global key/value kernel and cannot exactly
  represent pair-specific periodic images.

The smoke dataset and trainer in `platonic_transformers.datasets.mp20` and
`mains/main_mp20_smoke.py` provide a small CIF-backed periodic regression run.
The default CIF parser reads listed atom sites without symmetry expansion so
large local MP-style CIF directories can be tested quickly; pass
`parser="pymatgen"` or `--parser pymatgen` when full pymatgen parsing is
needed.
