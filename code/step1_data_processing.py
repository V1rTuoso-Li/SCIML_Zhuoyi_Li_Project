
import os, json, math, sys, csv, argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_csv(path):
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    elif 'time' in df.columns:
        ts = pd.to_datetime(df['time'], errors='coerce', utc=True)
    else:
        c = df.columns[0]
        ts = pd.to_datetime(df[c], errors='coerce', utc=True)
    df.index = ts
    df = df[~df.index.isna()]
    return df

def normalize_timezone(df, tz):
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    idx = df.index.tz_convert(tz)
    df = df.copy()
    df.index = idx
    return df

def rename_columns(df):
    cols = {c:c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    if 'outdoor' in df.columns:
        df = df.rename(columns={'outdoor':'toutf'})
    if 'outdoor_temp_f' in df.columns:
        df = df.rename(columns={'outdoor_temp_f':'toutf'})
    if 'indoor' in df.columns:
        df = df.rename(columns={'indoor':'tinf'})
    if 'indoor_temp_f' in df.columns:
        df = df.rename(columns={'indoor_temp_f':'tinf'})
    if 'u' in df.columns:
        df = df.rename(columns={'u':'u_raw'})
    if 'hvac' in df.columns:
        df = df.rename(columns={'hvac':'u_raw'})
    return df

def select_numeric(df):
    num = df.select_dtypes(include=[np.number]).copy()
    for c in ['toutf','tinf','u_raw']:
        if c not in num.columns:
            if c == 'u_raw':
                num[c] = np.nan
            else:
                num[c] = np.nan
    return num

def cap_outliers(s, lo, hi):
    x = s.copy()
    x = np.where(np.isnan(x), np.nan, x)
    x = np.clip(x, lo, hi)
    return pd.Series(x, index=s.index)

def interpolate_index(df, freq):
    idx = pd.date_range(df.index.min().floor(freq), df.index.max().ceil(freq), freq=freq, tz=df.index.tz)
    df = df.reindex(idx)
    df = df.interpolate(method='time', limit_direction='both')
    return df

def binarize_u(s, thresh=0.5):
    x = s.copy()
    x = np.where(np.isnan(x), 0, x)
    x = np.where(x>=thresh, 1, 0)
    return pd.Series(x, index=s.index).astype(int)

def build_calendar(i):
    z = pd.DataFrame(index=i)
    z['minute']=z.index.minute
    z['hour']=z.index.hour
    z['dow']=z.index.dayofweek
    z['dom']=z.index.day
    z['month']=z.index.month
    z['is_weekend']=((z['dow']>=5).astype(int))
    return z

def lagmat(s, L, prefix):
    out = {}
    for k in range(1, L+1):
        out[f'{prefix}_lag{k}'] = s.shift(k)
    return pd.DataFrame(out, index=s.index)

def process(path_in, tz, freq, L, outdir):
    df0 = load_csv(path_in)
    df0 = rename_columns(df0)
    df0 = select_numeric(df0)
    df0 = normalize_timezone(df0, tz)
    df0 = df0.sort_index()
    df0['toutf'] = cap_outliers(df0['toutf'], -40, 130)
    df0['tinf'] = cap_outliers(df0['tinf'], 30, 110)
    df0['u_raw'] = cap_outliers(df0['u_raw'], 0, 1)
    df1 = interpolate_index(df0, freq)
    df1['u'] = binarize_u(df1['u_raw'].fillna(0))
    cal = build_calendar(df1.index)
    xlags_tin = lagmat(df1['tinf'], L, 'tin')
    xlags_tout = lagmat(df1['toutf'], L, 'tout')
    xlags_u = lagmat(df1['u'], L, 'u')
    feats = pd.concat([cal, xlags_tin, xlags_tout, xlags_u], axis=1)
    y = df1[['tinf','toutf','u']].copy()
    y['t_next'] = y['tinf'].shift(-1)
    feats = feats.iloc[L:-1].copy()
    y = y.iloc[L:-1].copy()
    meta = {
        'tz': tz,
        'freq': freq,
        'L': L,
        'columns': list(df1.columns),
        'index_start': str(df1.index.min()),
        'index_end': str(df1.index.max())
    }
    _ensure_dir(outdir)
    clean_path = os.path.join(outdir,'processed_data.csv')
    feats_path = os.path.join(outdir,'features_window{:d}.csv'.format(L))
    target_path = os.path.join(outdir,'targets.csv')
    meta_path = os.path.join(outdir,'processing_meta.json')
    df1.to_csv(clean_path, index=True)
    feats.to_csv(feats_path, index=True)
    y.to_csv(target_path, index=True)
    with open(meta_path,'w') as f:
        json.dump(meta, f, indent=2)
    print(clean_path)
    print(feats_path)
    print(target_path)
    print(meta_path)

def split_train_val_test(index, ratios):
    n = len(index)
    a = int(n*ratios[0])
    b = int(n*ratios[1]) + a
    tr = index[:a]
    va = index[a:b]
    te = index[b:]
    return tr, va, te

def save_splits(feats, outdir, ratios=(0.6,0.2,0.2)):
    tr, va, te = split_train_val_test(feats.index.values, ratios)
    p = os.path.join(outdir,'splits.json')
    with open(p,'w') as f:
        json.dump({'train':list(map(str,tr)), 'val':list(map(str,va)), 'test':list(map(str,te))}, f, indent=2)
    print(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='data/kbwi_2025.csv')
    ap.add_argument('--tz', type=str, default='America/New_York')
    ap.add_argument('--freq', type=str, default='5min')
    ap.add_argument('--lags', type=int, default=12)
    ap.add_argument('--outdir', type=str, default='artifacts/step1')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.outdir), exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)
    process(args.input, args.tz, args.freq, args.lags, args.outdir)
    feats = pd.read_csv(os.path.join(args.outdir,'features_window{:d}.csv'.format(args.lags)), index_col=0, parse_dates=True)
    save_splits(feats, args.outdir)

if __name__ == '__main__':
    main()
