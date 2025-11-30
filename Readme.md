# Project Assets Overview

This repository documents the artifacts for a three-step pipeline that implements **ML-based Predictive Control for Central HVAC in U.S. 

---

## Step 1 — Data Processing

**Purpose:** Ingest raw NOAA Global-Hourly data, resample to 5-minute cadence, synthesize/align indoor temperature and ON/OFF signals, and build supervised learning datasets.

### Inputs
- `data/kbwi_2025.csv.csv`  
  Raw NOAA Global-Hourly source (BWI station, 2025). Original input to Step 1.

### Outputs
- `data/processed_data.csv`  
  5-minute, time-aligned series with outdoor temperature `ToutF`, synthesized indoor temperature `Tin`, and binary compressor command `u`.
- `data/features_window12.csv`  
  Feature matrix for forecasting with lag window `L=12` (5-min steps). Includes `Tin_lag_*`, `ToutF_lag_*`, `u_lag_*`.
- `data/targets.csv`  
  Forecast targets aligned to the future: `Tin_h3` (+15 min), `Tin_h6` (+30 min), `Tin_h12` (+60 min).
- `data/processing_meta.json`  
  Metadata from Step 1 (frequency, lags, horizons, source filename, generation timestamp).
- `data/splits.json`  
  Contiguous train/val/test segmentation indexes and their timestamp spans (reused by Steps 2–3).

---

## Step 2 — System Identification & Forecasting

**Purpose:** Identify a one-state RC surrogate for the plant and train a compact multi-horizon indoor temperature forecaster; compute a residual-based uncertainty proxy for robust MPC.

### Outputs
- `models/rc_params.json`  
  Identified one-state RC parameters (`a, b, c, d`) for the plant surrogate.
- `models/forecaster_params.npz`  
  Trained multi-horizon forecaster weights (NumPy archive) for 3/6/12-step indoor temperature prediction.
- `models/sigma_series.csv`  
  Residual-based uncertainty proxy time series `sigma_t` used for MPC constraint tightening.

---

## Step 3 — MPC & Benchmarks

**Purpose:** Run MILP–MPC with robust tightening, plus Deadband (DB) and Look-ahead (LA) baselines; produce schedules and comparison metrics.

### Typical Outputs
- `outputs/mpc_schedule.csv`  
  Receding-horizon binary control schedule `u(t)` with timestamps (primary MPC result).
- `outputs/la_schedule.csv` *(if generated)*  
  Look-ahead heuristic control schedule.
- `outputs/db_schedule.csv` *(if generated)*  
  Deadband thermostat control schedule.
- `outputs/metrics_summary.json` *(if generated)*  
  Aggregated comparison metrics (comfort minutes-in-band, duty, starts/hour, average solve time, fallback rate).

---

## Notes

- File paths are organized for downstream reproducibility across steps.
- Step 2 consumes Step 1 outputs (`features_window12.csv`, `targets.csv`, `splits.json`); Step 3 consumes Step 2 outputs (`rc_params.json`, `forecaster_params.npz`, `sigma_series.csv`) and Step 1’s processed time series where applicable.
