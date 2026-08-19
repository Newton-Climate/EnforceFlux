# FLEXPART test-met fixture

`AVAILABLE` indexes the two GRIB files that ship in the FLEXPART submodule at
`flexpart/tests/testdata/`. Configs point at that directory for `meteo_dir`
and at this `AVAILABLE` file, so FLEXPART configs run offline once
`make install-flexpart` has cloned the submodule.

Coverage: 2009-01-01 00:00Z and 06:00Z (global). Configs targeting FLEXPART
use these dates.
