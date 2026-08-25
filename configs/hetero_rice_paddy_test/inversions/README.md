# July 22 rice-paddy sensor inversions

These configs use the completed nature run
`source_heterogeneity_les_rice_paddy_l200_cv2p0_wind3_surface` and its diagnosed LES
turbulence to compare open-path (OP) and point-sensor networks with 1–4
sensors.

## Geometry

- Sensor height: 2 m AGL.
- The LES-diagnosed wind travels toward 215.02 degrees.
- The sensor transect is perpendicular to that wind at 305.02 degrees.
- The transect center is 20 m beyond the projected downwind edge of the
  1,000 m by 1,000 m emissions field: `(x, y) = (-411.13, -586.70) m`.
- Every OP network covers the same 1,000 m transect. N=1, 2, 3, and 4 divide
  it into adjacent paths of 1,000, 500, 333.33, and 250 m, respectively.
- Point sensors are placed at the centers of those same N segments. This is
  the cleanest field-realistic comparison because both technologies cover the
  same crosswind plume fence without placing multiple point sensors on one
  streamline.

The nature field is imposed as a mass-conserving 2-D bottom-boundary flux;
this avoids the low bias caused by the former 25 m-deep Gaussian releases and
matches bLS's surface-source physics. The inverse state is one scalar,
`Q_total`. A uniform spatial template maps
that total onto the source grid, avoiding use of the true heterogeneous field
shape as oracle information. The bLS operator uses the final LES-diagnosed
turbulence interval; observations are sampled from the MicroHH nature field
and time-averaged before the nonnegative inversion.

The 2,700 s MicroHH spin-up is excluded upstream when the native cross-sections
are canonicalized. The nature `concentration.nc` records
`spinup_discarded_s: 2700` and contains only iterations 2700–5400. The bLS
meteorology similarly uses the final runtime-sized LES column interval.

The 2 m resolved sonic covariance alone gives `u* = 0.065 m/s`; that excludes
the SGS stress that carries most momentum flux near an LES wall. The inverse
operators instead use MicroHH's SGS-inclusive column diagnostics averaged over
the post-spinup runtime: `u* = 0.29413 m/s` and `L = -28.742 m`. Resolved 10 Hz
fluctuations supply wind direction and mean wind. Because resolved variances
collapse near an LES wall, total component variability uses bLS surface-layer
ratios `sigma_u/u*=2.5`, `sigma_v/u*=2.0`, and `sigma_w/u*=1.25`. Flux
inversions include an absolute `25,000 ng m-3` error floor plus 30% relative
transport error at a `0.03 kg/s` flux scale. This combined representation
error prevents the most sensitive path segment from receiving unjustified
weight. The total-flux prior is deliberately weak (`variance = 1 kg2/s2`).

## Running one inversion

For example, the two-open-path case is:

```bash
enforceflux dispersion --config configs/hetero_rice_paddy_test/inversions/n2_op_operator.yaml
enforceflux instrument --config configs/hetero_rice_paddy_test/inversions/n2_op_instrument.yaml
enforceflux flux --config configs/hetero_rice_paddy_test/inversions/n2_op_flux.yaml
enforceflux analysis --config configs/hetero_rice_paddy_test/inversions/n2_op_analysis.yaml
```

Replace `n2_op` with `n1_op`, `n3_op`, `n4_op`, or the corresponding
`n1_point` through `n4_point` prefix.

## Practical point-sensor recommendation

The configured crosswind-fence layouts are recommended for this controlled
OP-versus-point inversion. For a real campaign, use a separate upwind
background monitor if available. If that monitor must count against a strict
four-sensor budget, use three downwind points (center and two flanks) plus one
upwind point; otherwise keep all four downwind positions configured here.
