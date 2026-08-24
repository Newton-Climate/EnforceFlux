# bLSmodelR — backward Lagrangian Stochastic transport

`enforceflux.blsmodelr` is a subprocess bridge to the R package
[bLSmodelR](https://github.com/ChHaeni/bLSmodelR) (Häni et al. 2018).
Backward LS is the community-standard forward model for field-scale
(10 m – 1 km) agricultural area-source inversion, and is used here as the
inversion transport operator for OSSEs with LES nature runs.

## Install

Requires R ≥ 4.0 on PATH.

macOS (recommended):
```bash
brew install --cask r
```

Debian/Ubuntu:
```bash
sudo apt-get install r-base
```

Then install the R packages once per R environment:
```bash
Rscript src/enforceflux/blsmodelr/r/install_bls.R
```

This installs `bLSmodelR` from GitHub plus its CRAN deps (`jsonlite`,
`data.table`, `sp`, `Rcpp`, `RcppArmadillo`, `remotes`).

Verify:
```bash
python -c "from enforceflux.blsmodelr import BlsWrapper; \
  BlsWrapper().check_rscript(); print('ok')"
```

## Contract

The Python wrapper serializes a `BlsRequest` (sensors, source polygons,
turbulence intervals, model params) to JSON and invokes
`r/run_bls.R --config … --out …`. The shim reads JSON in, writes long-form
CSV out (`sensor, source, interval, cxe, cxe_se, n_particles_used`).

The shim is the only file that touches bLSmodelR's API. See the
`BLSMODELR CALL` block in `r/run_bls.R` for the real `runbLS()` invocation;
the surrounding code is stable transport plumbing.

## Dry-run mode

`BlsWrapper({"dry_run": True})` (or `Rscript run_bls.R --dry-run`) skips
bLSmodelR and computes an analytic stub CxE proportional to
1 / (wind_speed × source–sensor distance). This lets you exercise the full
pipeline on machines without R, or with R but without bLSmodelR installed.

## Testing

```bash
pytest tests/blsmodelr/
```

The plumbing tests inject a fake `Rscript` and require no R install.
A separate smoke test runs the real R shim in dry-run mode; it is skipped
automatically when Rscript is not on PATH.
