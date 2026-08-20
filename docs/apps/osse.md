# `osse` — legacy end-to-end driver

**What it does.** Runs a full simple OSSE — source → instrument → transport →
inversion → metrics — from a single JSON config, in one shot. Written before
the per-stage pipeline existed.

**Kept for backward compatibility only.** For new work use the per-stage
apps: [dispersion](dispersion.md) → [instrument](instrument.md) →
[flux](flux.md) → [analysis](analysis.md). They are more transparent, cache
intermediate results, and are what all current examples target.

## Run it

```bash
enforceflux osse --config examples/quickstart_config.json
```

Prints a small summary: prior/posterior means and stds, Jacobian rank, null
space dimension, condition number.

## When you might still want it

- **Absolute minimal quickstart** — one command from a single JSON is easier
  to send someone than a four-step pipeline.
- **Reproducing pre-refactor experiments** whose configs are checked into
  history in the old JSON shape.

Everything the legacy driver does is available as a subset of the per-stage
pipeline. If you find yourself editing this app, port your workflow to the
per-stage YAMLs under [`configs/main/`](../../configs/main/) instead.
