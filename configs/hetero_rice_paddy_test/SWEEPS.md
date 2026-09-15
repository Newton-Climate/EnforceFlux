# Rice-paddy L, CV, seed, and sensor-count sweeps

The sweep is defined in code, in `sweep.py`. There are no generated config
files: each stage config is rendered in memory from `les_l200_cv2p0_wind3.yaml`
and the `inversions/` templates, written to a throwaway file for the length of
one `enforceflux` call, and deleted. Each run directory keeps the resolved
config as `config.snapshot.yaml`.

The default grid is Cartesian:

- Correlation length `L`: 100, 150, 200, 250, 300, 400, and 500 m.
- Coefficient of variation `CV`: 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, and 2.0.
- Emission-field seed: 0 through 19.
- Sensor count `n`: 1, 2, 3, and 4.
- Network: open path (`op`) and matched point sensor (`point`).

That is 980 source fields and 7,840 inversions. It contains the original
3 x 3 grid (L = 100, 250, 500 m by CV = 0.5, 1, 2), whose seeds 0-7 were run
first and 8-19 on 2026-09-14. The mesh points between them were added on
2026-09-15. Rendering the original grid reproduces the old
`generate_sweeps.py` output byte for byte, so existing runs are recognized as
complete and not rerun.

Seeds are shared across (L, CV): seed *k* feeds the same normal draws to every
condition (common random numbers), but only among conditions that use the same
sampler. On this 25 x 25, 40 m grid, circulant embedding is not positive
semi-definite for L >= 350 m. The generator then falls back to padded
Cholesky, and the same seed maps to a different field. So L = 400 and 500 m
share seeds with each other but not with L <= 300 m.

## Nature cases are evaluated, not integrated

A seeded nature case is *not* an LES run. It is rendered exactly as it would be
integrated — same grid, forcing, restart, and surface boundary condition — and
then evaluated through the tagged-tracer transport operator built by
`notebooks/hetero_experiments/run_les_tagged_operator.py`:

    concentration = H . e

where `e` is the case's own kinematic surface flux at the 196 tagged source
cells. This costs about six seconds instead of hours, which is what makes the
seed dimension affordable. Deleting the `microhh.operator_npz` key from a
rendered config (`--write`, below) runs that same realization as a real LES, so
any realization worth checking can be checked.

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
varies with L, CV, or seed. `sweep.py` therefore builds it once and copies it
into every other combination, preferring an operator already built by the
pre-seed sweep. Only the truth files (`truth_field.nc`, `basis_mapping.npz`)
differ per combination.

## Running

Run or resume the default grid (completed stages are skipped). One source
field with its eight inversions takes about 30 s:

```bash
python configs/hetero_rice_paddy_test/sweep.py --nature --inversions
```

Split it across processes. Shard 0 holds the bLS donor combination:

```bash
python configs/hetero_rice_paddy_test/sweep.py --nature --inversions --shard 0 --nshards 4
```

Run a subset, or a grid other than the default:

```bash
python configs/hetero_rice_paddy_test/sweep.py --nature --inversions --L 150,300 --cv 0.75 --seeds 0-19
```

Report how much of the grid is done:

```bash
python configs/hetero_rice_paddy_test/sweep.py --count
```

Write the rendered configs to disk, to inspect them or run one by hand:

```bash
python configs/hetero_rice_paddy_test/sweep.py --write /tmp/rice_cfgs --L 500 --cv 1.0 --seeds 3
enforceflux dispersion --config /tmp/rice_cfgs/nature_sweep/les_l500_cv1p0_s3_wind3.yaml
```

Relative paths in the configs assume they sit one directory below
`configs/hetero_rice_paddy_test/`, as `sweep.py`'s own scratch directory does.
Configs written elsewhere with `--write` are for reading. To run them, write
them to a directory at that depth, for example
`configs/hetero_rice_paddy_test/scratch`.
