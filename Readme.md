Data (raw & processed)

data/kbwi_2025.csv
Raw NOAA Global-Hourly source used to derive outdoor temperature for 2025 (BWI station). Original input to Step 1.

data/processed_data.csv
5-minute, time-aligned series with outdoor temperature (ToutF), synthesized indoor temperature (Tin), and binary compressor command (u).

data/features_window12.csv
Feature matrix for forecasting with lag window L=12 (5-min steps). Includes Tin_lag_*, ToutF_lag_*, u_lag_*.

data/targets.csv
Forecast targets aligned to the future: Tin_h3 (15 min), Tin_h6 (30 min), Tin_h12 (60 min).

data/processing_meta.json
Metadata from Step 1 (frequency, lags, horizons, source filename, generation timestamp).

data/splits.json
Contiguous train/val/test segmentation indexes and their timestamp spans (used by Steps 2–3).

Models & parameters (Step 2 outputs)

models/rc_params.json
Identified one-state RC model parameters (a,b,c,d) for the plant surrogate.

models/forecaster_params.npz
Trained multi-horizon forecaster weights (NumPy archive) for 3/6/12-step indoor temperature prediction.

models/sigma_series.csv
Residual-based uncertainty proxy time series (sigma_t) used for robust constraint tightening in MPC
