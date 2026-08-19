# Paired point/OP observability experiment

This experiment compares point sensors with 400 m open-path beams using the
same network centres. Each point is located at the midpoint of its paired OP
beam, and every OP beam is perpendicular to the LES wind direction.

The driver creates stage configurations in a temporary directory. Only the six
templates in this directory are persistent.

```bash
# Validate the matrix and required LES nature runs.
python notebooks/hetero_experiments/run_paired_observability_experiment.py

# Run one L/CV/layout/network cell for point and OP, using both inversions.
python notebooks/hetero_experiments/run_paired_observability_experiment.py --pilot

# Run the configured 450 sensor/operator cells and 900 inversions.
python notebooks/hetero_experiments/run_paired_observability_experiment.py --full

# Run one L/CV/network slice, for example the most heterogeneous N=2 case.
python notebooks/hetero_experiments/run_paired_observability_experiment.py \
  --full --L-source 200 --cv 2.0 --n-instruments 2

# Rebuild result, equality, and required-network CSV summaries.
python notebooks/hetero_experiments/run_paired_observability_experiment.py --summarize
```

The two inversion modes are:

- `total_uniform`: one domain-total state distributed with a uniform, non-oracle
  spatial template.
- `spatial_gp`: 18 shared coarse-grid states with an identical GP prior for
  point and OP networks; the posterior states are summed for total emissions.

`experiment.yaml` controls the L/CV/network/layout matrix, equivalence bounds,
and success criterion. The currently available LES nature runs contain source
seed 0 and the July 1, 2026 meteorology. Add completed nature runs before
extending `source_seeds` or `meteorology.cases`.
