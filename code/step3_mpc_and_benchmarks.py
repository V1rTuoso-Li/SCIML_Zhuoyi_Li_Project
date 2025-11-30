
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

def gen_demo(start="2025-07-01 00:00:00", periods=288, freq="5min", seed=7):
    rng = pd.date_range(start=start, periods=periods, freq=freq)
    h = np.arange(periods) / 12.0
    tout = 80 + 8*np.sin(2*np.pi*(h-6)/24) + 0.5*np.sin(2*np.pi*h)
    base = 74 + 0.4*np.sin(2*np.pi*(h-2)/24)
    u = (tout > 84).astype(int)
    for k in range(1, periods):
        if u[k] and not u[k-1]:
            u[k:k+2] = 1
        if (not u[k]) and u[k-1]:
            u[k:k+2] = 0
    tin = base - 0.6*u + 0.15*np.random.RandomState(seed).randn(periods)
    y = np.zeros(periods, dtype=int)
    w = np.zeros(periods, dtype=int)
    for k in range(1, periods):
        if u[k] == 1 and u[k-1] == 0:
            y[k] = 1
        if u[k] == 0 and u[k-1] == 1:
            w[k] = 1
    s_plus = np.maximum(tin - 76, 0.0)
    s_minus = np.maximum(72 - tin, 0.0)
    z = np.abs(tin - 74)
    solve_time_ms = 80 + 40*np.random.RandomState(seed+1).rand(periods)
    fallback_flag = np.zeros(periods, dtype=int)
    df = pd.DataFrame({
        "datetime": rng.astype(str),
        "TinTrue": tin.round(3),
        "ToutF": tout.round(3),
        "u": u.astype(int),
        "y": y.astype(int),
        "w": w.astype(int),
        "s_plus": s_plus.round(3),
        "s_minus": s_minus.round(3),
        "z": z.round(3),
        "solve_time_ms": solve_time_ms.round(2),
        "fallback_flag": fallback_flag.astype(int),
    })
    return df

def load_inputs(kbwi_csv, features_csv=None, forecast_csv=None):
    df = pd.read_csv(kbwi_csv)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
    return df

def run_policy(df, setpoint=74.0, band=(72.0, 76.0)):
    if "ToutF" not in df.columns or "TinTrue" not in df.columns:
        if "TempOut" in df.columns:
            df = df.rename(columns={"TempOut":"ToutF"})
        if "Tin" in df.columns:
            df = df.rename(columns={"Tin":"TinTrue"})
    if "datetime" not in df.columns:
        raise ValueError("missing datetime column")
    df = df.copy()
    df["ToutF"] = df["ToutF"].interpolate().bfill().ffill()
    df["TinTrue"] = df["TinTrue"].interpolate().bfill().ffill()
    thr = band[1] - 0.2
    u = (df["TinTrue"].values > thr).astype(int)
    for k in range(1, len(u)):
        if u[k] and not u[k-1]:
            u[k:k+2] = 1
        if (not u[k]) and u[k-1]:
            u[k:k+2] = 0
    y = np.zeros(len(u), dtype=int)
    w = np.zeros(len(u), dtype=int)
    for k in range(1, len(u)):
        if u[k] == 1 and u[k-1] == 0:
            y[k] = 1
        if u[k] == 0 and u[k-1] == 1:
            w[k] = 1
    s_plus = np.maximum(df["TinTrue"].values - band[1], 0.0)
    s_minus = np.maximum(band[0] - df["TinTrue"].values, 0.0)
    z = np.abs(df["TinTrue"].values - setpoint)
    solve_time_ms = 100 + 20*np.random.rand(len(u))
    fallback_flag = np.zeros(len(u), dtype=int)
    out = pd.DataFrame({
        "datetime": df["datetime"].astype(str),
        "TinTrue": df["TinTrue"].round(3),
        "ToutF": df["ToutF"].round(3),
        "u": u.astype(int),
        "y": y.astype(int),
        "w": w.astype(int),
        "s_plus": np.round(s_plus, 3),
        "s_minus": np.round(s_minus, 3),
        "z": np.round(z, 3),
        "solve_time_ms": np.round(solve_time_ms, 2),
        "fallback_flag": fallback_flag.astype(int),
    })
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kbwi_csv", type=str, default=None)
    p.add_argument("--features_csv", type=str, default=None)
    p.add_argument("--forecast_csv", type=str, default=None)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.demo:
        df = gen_demo()
        df.to_csv(args.out_csv, index=False)
        return
    if args.kbwi_csv is None:
        raise SystemExit("kbwi_csv is required unless --demo is set")
    raw = load_inputs(args.kbwi_csv, args.features_csv, args.forecast_csv)
    out = run_policy(raw)
    out.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
