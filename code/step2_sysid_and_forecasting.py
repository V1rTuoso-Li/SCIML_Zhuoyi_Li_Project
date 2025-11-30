
import os, json, argparse
import numpy as np
import pandas as pd

def ridge(X, y, lam, prior=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    if prior is not None:
        b = b + lam * prior
    w = np.linalg.solve(A, b)
    return w

def rc_design(df):
    T = df['tinf'].values[:-1]
    U = df['u'].values[:-1]
    Tout = df['toutf'].values[:-1]
    Y = df['tinf'].values[1:]
    X = np.column_stack([T, U, Tout, np.ones_like(T)])
    return X, Y

def whiteness(res):
    r = res - res.mean()
    ac1 = np.corrcoef(r[:-1], r[1:])[0,1] if len(r)>2 else 0.0
    return float(ac1)

def forecaster_fit(X_tr, y_tr, X_va, y_va, heads=(3,6,12), iters=400, lr=1e-2):
    d = X_tr.shape[1]
    H = len(heads)
    W = np.random.randn(d, H)*0.01
    b = np.zeros(H)
    for t in range(iters):
        pred = X_tr @ W + b
        e = pred - y_tr[:, :H]
        gW = X_tr.T @ e / len(X_tr)
        gb = e.mean(axis=0)
        W -= lr * gW
        b -= lr * gb
    return {'W':W, 'b':b, 'heads':list(heads)}

def forecaster_predict(params, X):
    W = params['W']
    b = params['b']
    return X @ W + b

def rolling_sigma(err_va, win=48):
    s = pd.Series(err_va)
    z = s.abs().rolling(win, min_periods=1).mean().values
    return z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_clean', type=str, default='artifacts/step1/processed_data.csv')
    ap.add_argument('--in_feats', type=str, default='artifacts/step1/features_window12.csv')
    ap.add_argument('--in_targets', type=str, default='artifacts/step1/targets.csv')
    ap.add_argument('--in_splits', type=str, default='artifacts/step1/splits.json')
    ap.add_argument('--outdir', type=str, default='artifacts/step2')
    ap.add_argument('--ridge', type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.in_clean, index_col=0, parse_dates=True)
    X_rc, Y_rc = rc_design(df)
    prior = np.array([0.9, -0.1, 0.05, 0.0])
    w = ridge(X_rc, Y_rc, args.ridge, prior=prior)
    res = Y_rc - (X_rc @ w)
    meta = {'a':w[0], 'b':w[1], 'c':w[2], 'd':w[3], 'r2':float(1 - (res**2).sum()/((Y_rc-Y_rc.mean())**2).sum()), 'ac1':whiteness(res)}
    with open(os.path.join(args.outdir,'rc_params.json'),'w') as f:
        json.dump(meta, f, indent=2)
    F = pd.read_csv(args.in_feats, index_col=0, parse_dates=True)
    T = pd.read_csv(args.in_targets, index_col=0, parse_dates=True)
    with open(args.in_splits,'r') as f:
        splits = json.load(f)
    idx = F.index.astype(str)
    m = pd.Series(np.arange(len(idx)), index=idx)
    tr = m.loc[splits['train']].values
    va = m.loc[splits['val']].values
    te = m.loc[splits['test']].values
    X_tr = F.values[tr]
    X_va = F.values[va]
    yh = ['tinf','toutf','u','t_next']
    Y = T[yh].copy()
    H = len([3,6,12])
    Y_shift = np.zeros((len(F), H))
    for j in range(H):
        Y_shift[:, j] = Y['t_next'].shift(j+1).values
    Y_shift = pd.DataFrame(Y_shift, index=F.index).iloc[:].values
    Y_tr = Y_shift[tr]
    Y_va = Y_shift[va]
    params = forecaster_fit(X_tr, Y_tr, X_va, Y_va, heads=(3,6,12), iters=600, lr=5e-3)
    P_va = forecaster_predict(params, X_va)
    err_va = (P_va - Y_va)[:,1]
    sig_va = rolling_sigma(err_va, win=48)
    np.savez(os.path.join(args.outdir,'forecaster_params.npz'), W=params['W'], b=params['b'], heads=np.array(params['heads']))
    pd.DataFrame({'err30':err_va, 'sigma30':sig_va}, index=F.index[va]).to_csv(os.path.join(args.outdir,'validation_sigma.csv'))
    df_sigma = pd.Series(sig_va, index=F.index[va]).reindex(F.index).ffill().bfill()
    df_sigma.to_csv(os.path.join(args.outdir,'sigma_series.csv'), header=['sigma'])
    print(os.path.join(args.outdir,'rc_params.json'))
    print(os.path.join(args.outdir,'forecaster_params.npz'))
    print(os.path.join(args.outdir,'validation_sigma.csv'))
    print(os.path.join(args.outdir,'sigma_series.csv'))

if __name__ == '__main__':
    main()
