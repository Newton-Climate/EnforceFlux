# Rice-paddy L, CV, seed, and sensor-count sweeps

The generated configs form this Cartesian grid:

- Correlation length `L`: 100, 250, and 500 m.
- Coefficient of variation `CV`: 0.5, 1.0, and 2.0.
- Emission-field seed: 0 through 7.
- Sensor count `n`: 1, 2, 3, and 4.
- Network: open path (`op`) and matched point sensor (`point`).

This produces 72 nature configs in `nature_sweep/` and 2,304 inversion-stage
configs in `inversion_sweep/` (72 source fields x 4 network sizes x 2 network
types x 4 stages). All configs retain the meteorology, placement, uncertainty,
and inversion choices from the latest `les_l200_cv2p0_wind3.yaml` and
`inversions/` templates.

## Nature cases are evaluated, not integrated

A seeded nature case is *not* an LES run. It is written exactly as it would be
integrated — same grid, forcing, restart, and surface boundary condition — and
then evaluated through the tagged-tracer transport operator built by
`notebooks/hetero_experiments/run_les_tagged_operator.py`:

    concentration = H . e

where `e` is the case's own kinematic surface flux at the 196 tagged source
cells. This costs about six seconds instead of hours, which is what makes the
seed dimension affordable. Deleting the `microhh.operator_npz` key from a
generated config runs that same realization as a real LES, so any realization
worth checking can be checked.

Two things follow from this and should be stated in any write-up:

- **Every seed shares one turbulence realization** — the operator's, warm
  started from the `l200_cv2p0` LES restart at 3,600 s. The seed dimension
  samples emission-field randomness only. Spread across seeds is conditional on
  a single eddy field, not a fully independent ensemble.
- **The operator costs a few percent.** Against the nine real LES runs it was
  validated on, `H . e` reproduces the 2 m concentration plane to about 4-5
  percent mass-weighted, with total mass within 1 percent. See
  `runs/source_heterogeneity_les_tagged_operator/validation.json`.

Seeded runs carry 45 cross-section frames (t = 3,660 to 6,300 s). The nine
original LES runs carry 46: they include a frame at the 3,600 s restart stamp
that is donor output left by the pre-fix `_seed_restart`, not their own. Pool
seeded runs with each other, not with the original nine.

## Operator reuse in the inversion stages

The bLS Jacobian depends on receptor geometry and met intervals, and neither
varies with L, CV, or seed. `run_sweeps.py` therefore builds it once and copies
it into every other combination, preferring an operator already built by the
pre-seed sweep. Only the truth files (`truth_field.nc`, `basis_mapping.npz`)
differ per combination.

## Running

Regenerate the configs after changing a template:

```bash
python configs/hetero_rice_paddy_test/generate_sweeps.py
```

Run or resume the complete suite (completed manifests are skipped):

```bash
python configs/hetero_rice_paddy_test/run_sweeps.py --nature --inversions
```

Run a single nature case:

```bash
enforceflux dispersion --config configs/hetero_rice_paddy_test/nature_sweep/les_l500_cv1p0_s3_wind3.yaml
```

Then run an inversion in stage order:

```bash
enforceflux dispersion --config configs/hetero_rice_paddy_test/inversion_sweep/l500_cv1p0_s3_n3_op_operator.yaml
enforceflux instrument --config configs/hetero_rice_paddy_test/inversion_sweep/l500_cv1p0_s3_n3_op_instrument.yaml
enforceflux flux --config configs/hetero_rice_paddy_test/inversion_sweep/l500_cv1p0_s3_n3_op_flux.yaml
enforceflux analysis --config configs/hetero_rice_paddy_test/inversion_sweep/l500_cv1p0_s3_n3_op_analysis.yaml
```
