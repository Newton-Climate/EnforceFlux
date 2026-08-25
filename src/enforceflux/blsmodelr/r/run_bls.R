#!/usr/bin/env Rscript
# Thin shim: JSON request in, CSV footprint out.
#
# Contract (must stay in sync with enforceflux.blsmodelr.wrapper):
#
# Input JSON (path passed via --config):
#   {
#     "sensors":   [{ "name": "S1", "x": 0.0, "y": 0.0, "z": 2.0 }, ...],
#     "sources":   [{ "name": "cell_000", "polygon_xy": [[x1,y1],[x2,y2],...] }, ...],
#     "intervals": [{ "id": "t0", "u_star": 0.35, "L": -50.0, "z0": 0.05,
#                     "wind_dir_deg": 225.0, "sd_u": 0.9, "sd_v": 0.8, "sd_w": 0.5,
#                     "z_ref": 4.0, "wind_speed": 3.2 }, ...],
#     "model":     { "n_particles": 50000, "max_traj_s": 200, "seed": 1 }
#   }
#
# Output CSV (path passed via --out):
#   sensor,source,interval,cxe,cxe_se,n_particles_used
#
# Units: cxe is concentration-per-emission [(kg m-3) / (kg m-2 s-1)] as
# returned by bLSmodelR's CxE column.
#
# The block marked "BLSMODELR CALL" is where the real runbLS() invocation
# goes. Everything above and below is stable transport plumbing.

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
})

parse_args <- function(argv) {
  a <- list(config = NULL, out = NULL, dry_run = FALSE)
  i <- 1
  while (i <= length(argv)) {
    if (argv[i] == "--config")   { a$config  <- argv[i + 1]; i <- i + 2 }
    else if (argv[i] == "--out") { a$out     <- argv[i + 1]; i <- i + 2 }
    else if (argv[i] == "--dry-run") { a$dry_run <- TRUE;   i <- i + 1 }
    else stop(sprintf("Unknown arg: %s", argv[i]))
  }
  if (is.null(a$config) || is.null(a$out))
    stop("Both --config and --out are required")
  a
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
req  <- fromJSON(args$config, simplifyVector = FALSE,
                 simplifyDataFrame = FALSE, simplifyMatrix = FALSE)

sensors   <- req$sensors
sources   <- req$sources
intervals <- req$intervals
model     <- req$model %||% list(n_particles = 50000L, max_traj_s = 200L, seed = 1L)
`%||%` <- function(a, b) if (is.null(a)) b else a

# --- BLSMODELR CALL --------------------------------------------------------
# Real path:
#   library(bLSmodelR)
#   sensor_df   <- rbindlist(lapply(sensors,   as.data.table))
#   source_list <- lapply(sources, function(s) {
#     poly <- do.call(rbind, s$polygon_xy)
#     list(name = s$name, polygon = poly)
#   })
#   interval_df <- rbindlist(lapply(intervals, as.data.table))
#   input <- genInputList(Sensors = sensor_df,
#                         Sources = source_list,
#                         Interval = interval_df,
#                         Model = list(N = model$n_particles,
#                                      MaxFetch = model$max_traj_s))
#   res <- runbLS(input, ncores = 1)
#   # res$CE is a data.table with columns Sensor, Source, Interval, CE, CE_se
#   out <- data.table(sensor = res$CE$Sensor, source = res$CE$Source,
#                     interval = res$CE$Interval, cxe = res$CE$CE,
#                     cxe_se = res$CE$CE_se,
#                     n_particles_used = model$n_particles)
#
# Until the real call is wired in, produce a deterministic stub so the
# Python-side plumbing (subprocess, JSON marshalling, CSV parsing) can be
# tested end-to-end without a live bLSmodelR install. --dry-run also takes
# this branch explicitly.
use_stub <- args$dry_run || !requireNamespace("bLSmodelR", quietly = TRUE)

if (use_stub) {
  rows <- list()
  for (s in sensors) for (src in sources) for (iv in intervals) {
    # Stub CxE ~ 1/(u * distance), rough enough to sanity-check plumbing.
    poly <- do.call(rbind, lapply(src$polygon_xy, unlist))
    cx <- mean(poly[, 1]); cy <- mean(poly[, 2])
    sx <- as.numeric(s$x); sy <- as.numeric(s$y)
    d  <- max(sqrt((sx - cx)^2 + (sy - cy)^2), 1.0)
    u  <- max(as.numeric(iv$wind_speed %||% 3.0), 0.1)
    npart <- as.integer(model$n_particles)
    rows[[length(rows) + 1]] <- data.table(
      sensor = as.character(s$name), source = as.character(src$name),
      interval = as.character(iv$id),
      cxe = 1.0 / (u * d), cxe_se = 0.05 / (u * d),
      n_particles_used = npart
    )
  }
  out <- rbindlist(rows)
} else {
  suppressPackageStartupMessages(library(bLSmodelR))
  # bLS switches to L'Ecuyer-CMRG internally. Initialize that same generator
  # after package loading (which can alter RNG state) and immediately before
  # constructing/running the stochastic trajectory model.
  RNGkind("L'Ecuyer-CMRG")
  set.seed(as.integer(model$seed %||% 1L))
  # --- Sensors --------------------------------------------------------------
  # bLSmodelR's genSensors takes NAMED arguments where each name becomes the
  # Sensor Name; the value is a list starting with the geometry code ("p" for
  # point) then x/y/z. Build the arg list and splice with do.call().
  sensor_names <- vapply(sensors, function(s) as.character(s$name), character(1))
  sensor_args <- setNames(
    lapply(sensors, function(s) list(
      "p", x = as.numeric(s$x), y = as.numeric(s$y), z = as.numeric(s$z)
    )),
    sensor_names
  )
  sensors_obj <- do.call(genSensors, sensor_args)

  # --- Sources --------------------------------------------------------------
  # Same idiom: one named argument per source, geometry code "p" for polygon,
  # then x/y vectors of the polygon vertices.
  src_names <- vapply(sources, function(s) as.character(s$name), character(1))
  source_args <- setNames(
    lapply(sources, function(s) {
      poly <- do.call(rbind, lapply(s$polygon_xy, unlist))
      list("p", x = as.numeric(poly[, 1]), y = as.numeric(poly[, 2]))
    }),
    src_names
  )
  sources_obj <- do.call(genSources, source_args)

  # --- Intervals ------------------------------------------------------------
  # One genInterval() call per JSON interval; row-bind into one Interval object.
  sensor_names_joined <- paste(sensor_names, collapse = ",")
  source_names_joined <- paste(src_names, collapse = ",")
  # bLSmodelR requires class 'Interval'; data.table's rbindlist drops it, so
  # use base rbind which preserves S3 classes.
  interval_list <- lapply(intervals, function(iv) {
    ustar <- as.numeric(iv$u_star)
    # σ_u/u*, σ_v/u*, σ_w/u* — bLS expects DIMENSIONLESS ratios. If the JSON
    # gives dimensional sd_u [m/s], divide by u_star; guard against tiny u*.
    ustar_safe <- max(ustar, 0.05)
    sUu <- if (is.null(iv$sd_u)) 2.5 else as.numeric(iv$sd_u) / ustar_safe
    sVu <- if (is.null(iv$sd_v)) 2.0 else as.numeric(iv$sd_v) / ustar_safe
    sWu <- if (is.null(iv$sd_w)) 1.25 else as.numeric(iv$sd_w) / ustar_safe
    # Sanity clamp: MOST-consistent similarity constants have Cor(u,w) < 1,
    # which requires sUu * sWu > ~1 for u*=1; enforce a floor.
    sUu <- max(sUu, 1.5); sVu <- max(sVu, 1.2); sWu <- max(sWu, 0.8)
    genInterval(
      Ustar = ustar, L = as.numeric(iv$L),
      Zo = as.numeric(iv$z0), WD = as.numeric(iv$wind_dir_deg),
      sUu = sUu, sVu = sVu, sWu = sWu, z_sWu = as.numeric(iv$z_ref),
      N0 = as.integer(model$n_particles),
      # MaxFetch is METRES of alongwind tracking distance, not seconds.
      # Negative → bLS auto-optimizes to max(sensor-source dist) + |value|.
      MaxFetch = -as.numeric(model$max_traj_s),
      SensorNames = sensor_names_joined,
      SourceNames = source_names_joined
    )
  })
  interval_dt <- do.call(rbind, interval_list)
  # Preserve JSON interval ids so the Python side can join back cleanly.
  interval_dt$rn <- as.character(seq_len(nrow(interval_dt)))
  json_ids <- vapply(intervals, function(iv) as.character(iv$id), character(1))
  id_map <- setNames(json_ids, interval_dt$rn)

  input_list <- genInputList(sensors_obj, sources_obj, interval_dt)

  cat_dir <- tempfile("bls_cat_", tmpdir = dirname(args$out))
  dir.create(cat_dir, showWarnings = FALSE)
  # `ncores`: 0 (default) → auto-detect all physical cores on this host so the
  # operator saturates the machine without the caller having to know how many
  # cores exist. Any positive int pins the worker count.
  ncores_req <- as.integer(model$ncores %||% 0L)
  ncores <- if (ncores_req > 0L) ncores_req else max(parallel::detectCores(logical = FALSE), 1L)
  res <- runbLS(input_list, Cat.Path = cat_dir, ncores = ncores,
                show_progress = FALSE, asDT = TRUE)

  out <- data.table(
    sensor   = as.character(res$Sensor),
    source   = as.character(res$Source),
    interval = unname(id_map[as.character(res$rn)]),
    cxe      = as.numeric(res$CE),
    cxe_se   = as.numeric(res$CE_se),
    n_particles_used = as.integer(model$n_particles)
  )
  # Some (sensor, source, interval) triples may be missing from the result
  # when bLS returns no trajectories inside the source; back-fill with 0 so
  # the Python-side reshape does not raise "missing combination".
  full <- CJ(sensor = sensor_names, source = src_names, interval = json_ids,
             unique = TRUE, sorted = FALSE)
  out <- merge(full, out, by = c("sensor","source","interval"), all.x = TRUE)
  out[is.na(cxe), cxe := 0.0]
  out[is.na(cxe_se), cxe_se := 0.0]
  out[is.na(n_particles_used), n_particles_used := as.integer(model$n_particles)]
}
# ---------------------------------------------------------------------------

fwrite(out, args$out)
cat(sprintf("wrote %d rows to %s\n", nrow(out), args$out))
