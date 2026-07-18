"""
============================================================
 Nowcasting & Forecasting Italian GDP
------------------------------------------------------------
 Pipeline 
   STEP 1  data loading & cleaning  -> clean monthly panel
   STEP 2  mixed-frequency DFM + Kalman filter (MLE) -> nowcast
   STEP 3  VAR(2) & Minnesota BVAR(4) -> quarterly forecasts,
           anchored on the nowcast, COVID handled with dummies
   STEP 4  pseudo-real-time out-of-sample evaluation
           (RMSFE vs AR(1)/RW, Diebold-Mariano, with/without 2020)
   STEP 5  the five deliverable plots
   STEP 6  saved outputs + final summary

 Data: FRED (GDP, exports, unemployment, HICP, 3m rate) and
 Eurostat (industrial production, retail, confidence surveys),
 raw CSVs in RAW/
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # save figures to file, no blocking window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import invwishart, norm
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)                           # reproducible random fill of missing values

# ============================================================
#  CONFIG — every choice of the project in one place
# ============================================================

RAW_DIR      = Path("RAW")                  # raw CSVs
SAMPLE_START = "2000-01"                    # estimation sample starts here
EVAL_START   = "2012-01-01"                 # out-of-sample evaluation starts here
P_VAR, P_BVAR, H = 2, 4, 4                  # VAR lags, BVAR lags, forecast horizon
LAM = dict(lam1=0.2, lam2=0.5, lam3=1.0, lam4=100.0)   # Minnesota hyperparameters (§7.4.3)
COVID_QUARTERS = ("2020Q1", "2020Q2", "2020Q3")        # Italy locked down in March 2020
PUB_LAGS = {"ip": 2, "retail": 2, "exports": 2,        # publication lags in months:
            "unemp": 1, "bci": 0, "cci": 0, "gdp": 2}  # surveys immediate, hard data late
W_AGG = np.array([1, 2, 3, 2, 1]) / 3.0     # mixed-frequency aggregation weights (Ch.2/5)

# ============================================================
#  SHARED HELPERS
# ============================================================

def build_X(data, p, dummies=None):
    """Regressor matrix: constant + p lags of all variables (+ optional dummies)."""
    T = len(data)
    cols = [np.ones((T - p, 1))] + [data[p - l:T - l] for l in range(1, p + 1)]
    if dummies is not None:
        cols.append(dummies[p:])
    return np.hstack(cols)

def minnesota_prior(data, p, k, n, lam1, lam2, lam3, lam4, levels, n_dummies):
    """Minnesota prior (§7.4.3): mean B0 and diagonal prior variance, vec by equation.
       Scale factors s_i = residual std of a univariate AR(p) (unit-free shrinkage)."""
    T_all = len(data); T = T_all - p
    s = np.zeros(n)
    for i in range(n):
        yi = data[p:, i]
        Xi = np.hstack([np.ones((T, 1))] +
                       [data[p - l:T_all - l, i:i + 1] for l in range(1, p + 1)])
        s[i] = np.std(yi - Xi @ np.linalg.lstsq(Xi, yi, rcond=None)[0])
    B0 = np.zeros(k * n); hdiag = np.zeros(k * n)
    for i in range(n):
        base = i * k
        hdiag[base] = (lam1 * lam4) ** 2                  # constant: essentially free
        col = 1
        for l in range(1, p + 1):
            for j in range(n):
                if j == i:
                    hdiag[base + col] = (lam1 / l ** lam3) ** 2
                    if l == 1:                            # own first lag: 1 if level, 0 if growth
                        B0[base + col] = 1.0 if levels[i] else 0.0
                else:
                    hdiag[base + col] = (lam1 * lam2 * s[i] / s[j] / l ** lam3) ** 2
                col += 1
        for _ in range(n_dummies):
            hdiag[base + col] = 10.0 ** 2; col += 1       # dummies: loose prior, mean 0
    return B0, hdiag, s

# ============================================================
#  STEP 1 — DATA LOADING & CLEANING
# ============================================================

def load_fred(fname):
    """FRED CSV: 'observation_date' + one value column."""
    s = pd.read_csv(RAW_DIR / fname, parse_dates=[0], na_values=".")
    s = s.set_index(s.columns[0]).iloc[:, 0]
    s.index.name = "date"
    return s

def load_eurostat(fname, s_adj="Seasonally and calendar adjusted data",
                  nace_start=None, indic=None):
    """Eurostat 'linear' CSV (long format), filtered to a single series."""
    df = pd.read_csv(RAW_DIR / fname)
    if s_adj is not None:      df = df[df["s_adj"] == s_adj]
    if nace_start is not None: df = df[df["nace_r2"].str.startswith(nace_start)]
    if indic is not None:      df = df[df["indic"] == indic]
    df = df.dropna(subset=["OBS_VALUE"])
    s = pd.Series(df["OBS_VALUE"].values,
                  index=pd.PeriodIndex(df["TIME_PERIOD"], freq="M").to_timestamp()).sort_index()
    s.index.name = "date"
    return s

# raw series (levels, as published)
fred_files = {"gdp": "PIL.csv", "exports": "Export copia.csv",
              "unemp": "disoccupazione.csv", "hicp": "Inflazione.csv",
              "rate3m": "tasso3m.csv"}
series = {name: load_fred(f) for name, f in fred_files.items()}
series["ip"]     = load_eurostat("prodindus_eu.csv", nace_start="Mining")
series["retail"] = load_eurostat("retail_eu.csv",    nace_start="Retail")
series["bci"]    = load_eurostat("FiduciaImprese_eu.csv",
                                 indic="Industrial confidence indicator",
                                 s_adj="Seasonally adjusted data, not calendar adjusted data")
series["cci"]    = load_eurostat("FiduciaConsu_eu.csv",
                                 indic="Consumer confidence indicator",
                                 s_adj="Seasonally adjusted data, not calendar adjusted data")

# transformation matched to each variable's nature (Ch.2)
transform = {"gdp": "logdiff", "ip": "logdiff", "retail": "logdiff", "exports": "logdiff",
             "hicp": "logdiff", "unemp": "diff", "rate3m": "diff",
             "bci": "level", "cci": "level"}

def make_stationary(s, how):
    if how == "logdiff": return 100 * np.log(s).diff()    # % growth
    if how == "diff":    return s.diff()
    if how == "level":   return s
    raise ValueError(how)

stat = {name: make_stationary(series[name], transform[name]) for name in series}
monthly = pd.DataFrame(stat).asfreq("MS").loc[SAMPLE_START:]   # common monthly calendar

out = Path("data/clean"); out.mkdir(parents=True, exist_ok=True)
monthly.to_csv(out / "clean_stationary.csv")
print("STEP 1 done — clean_stationary.csv", monthly.shape)


# ============================================================
#  STEP 2 — MIXED-FREQUENCY DFM + KALMAN FILTER (nowcast)
# ===========================================================

SOFT = [4, 5]        # bci, cci: soft indicators load on a 12-month sum of the factor (§5.1.2)

def matrices(theta):
    """From theta to the state-space matrices (course order: R, H, F, Q).
       State (29): [0:12] factor f_t..f_{t-11} | [12:17] GDP idio (5 lags)
                   | [17:29] monthly idio, AR(2) blocks (2 slots each)."""
    phi1, phi2 = theta[0], theta[1]          # factor AR(2)
    lam        = theta[2:8]                  # monthly loadings
    lam_g      = theta[8]                    # GDP loading
    psi        = theta[9:21].reshape(6, 2)   # monthly idio AR(2)
    psiG       = theta[21:23]                # GDP idio AR(2)
    sig2       = np.exp(theta[23:29])        # idio variances (log-parametrized > 0)
    sig2_g     = np.exp(theta[29])
    nf = 12
    m, n = nf + 5 + 12, 7
    F0, G0, E0 = 0, nf, nf + 5               # block starts: factor, GDP idio, monthly idio

    F = np.zeros((m, m))                     # ---- transition ----
    F[F0, F0] = phi1; F[F0, F0 + 1] = phi2
    for k in range(1, nf): F[F0 + k, F0 + k - 1] = 1.0        # factor lag conveyor
    F[G0, G0] = psiG[0]; F[G0, G0 + 1] = psiG[1]
    for k in range(1, 5): F[G0 + k, G0 + k - 1] = 1.0         # GDP-idio lag conveyor
    for i in range(6):
        b = E0 + 2 * i
        F[b, b] = psi[i, 0]; F[b, b + 1] = psi[i, 1]; F[b + 1, b] = 1.0

    Hm = np.zeros((n, m))                    # ---- observation ----
    for i in range(6):
        if i in SOFT: Hm[i, F0:F0 + nf] = lam[i]   # soft: 12-month sum of the factor
        else:         Hm[i, F0] = lam[i]           # hard: current factor only
        Hm[i, E0 + 2 * i] = 1.0                    # own idiosyncratic
    Hm[6, F0:F0 + 5] = lam_g * W_AGG               # GDP row: factor lags, triangular weights
    Hm[6, G0:G0 + 5] = W_AGG                       # GDP row: own idio lags, same weights

    R = np.eye(n)                            # base measurement noise (missing-data trick §5.2)
    Q = np.zeros((m, m))                     # state shocks only in "current" slots
    Q[F0, F0] = 1.0                          # factor shock variance fixed = 1 (identification)
    Q[G0, G0] = sig2_g
    for i in range(6): Q[E0 + 2 * i, E0 + 2 * i] = sig2[i]
    return R, Hm, F, Q

# --- observation matrix: columns aligned with the rows of H ---
clean = pd.read_csv("data/clean/clean_stationary.csv", index_col=0, parse_dates=[0])
cols  = ["ip", "retail", "exports", "unemp", "bci", "cci", "gdp"]
panel = clean[cols].copy()
panel["gdp"] = panel["gdp"].shift(2)         # quarterly GDP -> 3rd month of the quarter (§5.1.1)
mu_p, sd_p = panel.mean(), panel.std()
panelz = (panel - mu_p) / sd_p               # standardized panel
obs_mask = (~panelz.isna()).astype(float).values     # 1 = observed, 0 = missing (the s_t of §5.2)
yv = panelz.values.copy()
yv[np.isnan(yv)] = np.random.randn(int(np.isnan(yv).sum()))   # fill holes: values are irrelevant
dates = panelz.index

def kalman_filter(theta, yv, obs_mask, burn=12):
    """Kalman filter with the missing-data trick; the filter IS the likelihood (§4.4)."""
    R, Hm, F, Q = matrices(theta)
    m = F.shape[0]; T, n = yv.shape
    beta = np.zeros((m, 1)); P = np.eye(m)
    filt = np.zeros((T, m)); like = np.zeros(T)
    for t in range(T):
        idx = obs_mask[t]
        Hit = idx[:, None] * Hm              # zero the H-row of missing series
        Rit = np.diag(1.0 - idx)             # unit measurement variance where missing
        beta10 = F @ beta                    # prediction
        P10 = F @ P @ F.T + Q
        n10 = yv[t].reshape(-1, 1) - Hit @ beta10          # innovation (surprise)
        F10 = Hit @ P10 @ Hit.T + Rit; F10 = 0.5 * (F10 + F10.T)
        F10_inv = np.linalg.inv(F10)
        sign, logdet = np.linalg.slogdet(F10)
        like[t] = -0.5 * (n * np.log(2 * np.pi) + logdet
                          + (n10.T @ F10_inv @ n10).item())
        K = P10 @ Hit.T @ F10_inv            # Kalman gain
        beta = beta10 + K @ n10              # update
        P = P10 - K @ Hit @ P10
        filt[t] = beta.ravel()
    return like[burn:].sum(), filt           # burn-in: skip the diffuse start

def neg_loglik(theta):
    return -kalman_filter(theta, yv, obs_mask)[0]

# hand-picked starting values (as Homework1.m)
theta0 = np.array([
    0.5, 0.2,                                # factor AR(2)
    0.8, 0.6, 0.6, -0.5, 0.3, 0.3,           # monthly loadings (unemp negative)
    0.5,                                     # GDP loading
    0.2, 0.1, 0.2, 0.1, 0.2, 0.1,            # monthly idio AR(2), series 1-3
    0.2, 0.1, 0.2, 0.1, 0.2, 0.1,            # monthly idio AR(2), series 4-6
    0.2, 0.1,                                # GDP idio AR(2)
    *([np.log(0.5)] * 6), np.log(0.5),       # log-variances
])

print("STEP 2 — estimating the DFM by MLE (may take 1-2 min)...")
res = minimize(neg_loglik, theta0, method="BFGS", options={"maxiter": 1000})
print("  done. success:", res.success, " -loglik:", round(res.fun, 2))
theta_hat = res.x

ll, filt = kalman_filter(theta_hat, yv, obs_mask)
factor = filt[:, 0]

# sign normalization: the factor is identified up to sign -> align it with IP
mask_ip = ~panelz["ip"].isna().values
if np.corrcoef(factor[mask_ip], panelz["ip"].values[mask_ip])[0, 1] < 0:
    factor, filt = -factor, -filt

# factor-implied nowcast: GDP row applied to the factor block only
# (excludes the GDP idiosyncratic -> honest signal from the monthly indicators)
nowcast = (filt[:, 0:5] @ (theta_hat[8] * W_AGG)) * sd_p["gdp"] + mu_p["gdp"]
gdp_real = panel["gdp"]                       # realized growth (at quarter-end months)

# quick diagnostics
print("lam:", np.round(theta_hat[2:8], 3), " lam_g:", round(theta_hat[8], 3))
fac_sum12 = pd.Series(factor, index=dates).rolling(12).sum().values
for c in ["bci", "cci"]:
    mm = ~panelz[c].isna().values & ~np.isnan(fac_sum12)
    print(f"  {c} vs sum12(factor): {np.corrcoef(fac_sum12[mm], panelz[c].values[mm])[0, 1]:+.3f}")

# --- nowcast by quarter (read at quarter-end months) ---
nc = pd.Series(nowcast, index=dates)
qend_months = nc.index[nc.index.month.isin([3, 6, 9, 12])]
nowcast_table = pd.DataFrame({
    "nowcast_%":  nc[qend_months].values,
    "realized_%": gdp_real.reindex(qend_months).values,
}, index=[f"{d.year}Q{(d.month - 1)//3 + 1}" for d in qend_months])

print("\n=== GDP nowcast by quarter ===")
print(nowcast_table.tail(6).round(2))
cur_label = nowcast_table.index[-1]
nowcast_now = float(nowcast_table["nowcast_%"].iloc[-1])
print(f"\n>>> Nowcast {cur_label}: {nowcast_now:+.2f}%   (GDP not yet published)")

# ============================================================
#  STEP 3 — VAR(2) & BVAR(4): quarterly forecasts
# ============================================================
# COVID: the 2020 quarters (crash -13%, rebound +13%) inflate the residual
# variance ~8x. We report both "raw" (2020 in) and "dummy" (2020 absorbed).
# Dummies are 1 only in a COVID quarter and 0 in the future (macro4.m trick).

def make_covid_dummies(qindex, quarters=COVID_QUARTERS):
    """One dummy column per COVID quarter. Unlike the US case (macro4.m: Q2-Q3 only),
       Italy locked down in March 2020, so 2020Q1 is also affected and gets a dummy."""
    qmap = {"2020Q1": "2020-01-01", "2020Q2": "2020-04-01",
            "2020Q3": "2020-07-01", "2020Q4": "2020-10-01"}
    D = np.zeros((len(qindex), len(quarters)))
    for j, q in enumerate(quarters):
        ts = pd.Timestamp(qmap[q])
        if ts in qindex:
            D[qindex.get_loc(ts), j] = 1.0
    return D

def var_ols(df, p=2, dummies=None):
    """VAR(p) by OLS, all equations at once; optional COVID dummies."""
    data = df.values
    Y = data[p:]
    X = build_X(data, p, dummies)
    B = np.linalg.solve(X.T @ X, X.T @ Y)
    E = Y - X @ B
    Sigma = (E.T @ E) / (len(Y) - X.shape[1])
    return B, Sigma, X, Y

def var_forecast(B, df, h=4, p=2, current=None, n_dummies=0):
    """Iterate the VAR forward. Step 0 = current quarter: known values (nowcast,
       partial actuals) override the model where available. Future dummies = 0."""
    hist = [df.values[i] for i in range(-p, 0)]
    preds = []
    for step in range(h + 1):
        x = np.concatenate([[1.0]] + [hist[-l] for l in range(1, p + 1)] + [np.zeros(n_dummies)])
        yhat = x @ B
        if step == 0 and current is not None:
            known = ~np.isnan(current)
            yhat = yhat.copy(); yhat[known] = current[known]
        preds.append(yhat); hist.append(yhat)
    return np.array(preds)

def var_bands(B, Sigma, fc, p, n, labels, z=1.2816):
    """Analytic 80% bands: accumulate forecast-error variance via the companion
       matrix (S1 = Sigma, S2 = Sigma + Psi1 Sigma Psi1', ... as Varforecast.m)."""
    A = [B[1 + (l - 1) * n: 1 + l * n, :].T for l in range(1, p + 1)]
    C = np.zeros((n * p, n * p)); C[:n, :] = np.hstack(A)
    if p > 1: C[n:, :n * (p - 1)] = np.eye(n * (p - 1))
    J = np.zeros((n, n * p)); J[:n, :n] = np.eye(n)
    h = fc.shape[0] - 1
    gdp = fc[:, 0]
    p10 = np.zeros(h + 1); p90 = np.zeros(h + 1)
    p10[0] = p90[0] = gdp[0]                  # step 0 is conditioned -> no band
    MSE = np.zeros((n, n)); Cj = np.eye(n * p)
    for k in range(1, h + 1):
        Psi = J @ Cj @ J.T
        MSE = MSE + Psi @ Sigma @ Psi.T
        sd = np.sqrt(MSE[0, 0])
        p10[k] = gdp[k] - z * sd; p90[k] = gdp[k] + z * sd
        Cj = Cj @ C
    return pd.DataFrame({"p10": p10, "median": gdp, "p90": p90}, index=labels)

def bvar(df, p=4, levels=None, dummies=None, reps=5000, burn=1000, h=4,
         current=None, seed=0, **lam):
    """Minnesota BVAR by Gibbs sampling (as BVAR.m): alternate coefficient and
       Sigma draws, then simulate h steps ahead WITH fresh shocks -> forecast paths."""
    lamd = {**LAM, **lam}
    rng = np.random.default_rng(seed)
    data = df.values; n = data.shape[1]
    if levels is None: levels = [False] * n
    d = 0 if dummies is None else dummies.shape[1]
    Y = data[p:]
    X = build_X(data, p, dummies); T, k = X.shape
    B0, hdiag, s = minnesota_prior(data, p, k, n, lamd["lam1"], lamd["lam2"],
                                   lamd["lam3"], lamd["lam4"], levels, d)
    Hinv = np.diag(1.0 / hdiag)
    betaols = np.linalg.solve(X.T @ X, X.T @ Y).flatten(order="F")
    Sprior = np.eye(n); alpha = n + 1
    Sigma = np.cov(Y, rowvar=False); XtX = X.T @ X
    paths = []
    for rep in range(reps):
        Siginv = np.linalg.inv(Sigma)
        V = np.linalg.inv(Hinv + np.kron(Siginv, XtX))          # posterior covariance
        M = V @ (Hinv @ B0 + np.kron(Siginv, XtX) @ betaols)    # prior-OLS mix
        beta = M + np.linalg.cholesky(0.5 * (V + V.T)) @ rng.standard_normal(k * n)
        Bmat = beta.reshape(k, n, order="F")
        E = Y - X @ Bmat
        Sigma = invwishart.rvs(df=T + alpha, scale=E.T @ E + Sprior, random_state=rng)
        if rep >= burn:
            hist = [data[i].copy() for i in range(-p, 0)]
            cholS = np.linalg.cholesky(0.5 * (Sigma + Sigma.T)); path = []
            for step in range(h + 1):
                x = np.concatenate([[1.0]] + [hist[-l] for l in range(1, p + 1)] + [np.zeros(d)])
                yhat = x @ Bmat + cholS @ rng.standard_normal(n)
                if step == 0 and current is not None:
                    known = ~np.isnan(current)
                    yhat = yhat.copy(); yhat[known] = current[known]
                path.append(yhat); hist.append(yhat)
            paths.append(np.array(path))
    return np.array(paths)                    # (n_draws, h+1, n)

def gdp_fan(paths, labels):
    """Percentiles of the simulated GDP paths: the fan chart."""
    g = paths[:, :, 0]
    p10, p50, p90 = np.percentile(g, [10, 50, 90], axis=0)
    return pd.DataFrame({"p10": p10, "median": p50, "p90": p90}, index=labels)

# --- quarterly dataset: GDP is quarterly, the rest aggregated from monthly ---
q_gdp   = 100 * np.log(series["gdp"]).diff()
hicp_q  = series["hicp"].resample("QS").mean()
q_infl_raw = 100 * np.log(hicp_q).diff()
seas    = q_infl_raw.groupby(q_infl_raw.index.quarter).transform("mean")
q_infl  = q_infl_raw - seas + q_infl_raw.mean()      # HICP is NSA: remove seasonal means
q_unemp = series["unemp"].resample("QS").mean()
q_rate  = series["rate3m"].resample("QS").mean()
qdata = pd.DataFrame({"gdp": q_gdp, "infl": q_infl, "unemp": q_unemp,
                      "rate": q_rate}).dropna().loc[SAMPLE_START:]

D   = make_covid_dummies(qdata.index)
lev = [False, False, True, True]      # growth -> prior 0 (white noise); levels -> prior 1 (RW)

# --- forecast horizon + current-quarter conditioning ---
future_dates = pd.date_range(qdata.index[-1], periods=H + 2, freq="QS")[1:]
labels = [f"{d.year}Q{(d.month - 1)//3 + 1}" for d in future_dates]
current_q = future_dates[0]
current = np.array([
    nowcast_now,                       # gdp   -> DFM nowcast (computed in Step 2)
    q_infl.get(current_q, np.nan),     # infl  -> partial actual, if available
    q_unemp.get(current_q, np.nan),    # unemp -> partial actual, if available
    q_rate.get(current_q, np.nan),     # rate  -> NaN -> the VAR predicts it
])

# --- VAR(2): point forecast (raw vs dummy) + bands ---
B_raw, S_raw, _, _ = var_ols(qdata, p=P_VAR)
fc_raw = var_forecast(B_raw, qdata, h=H, p=P_VAR, current=current)[:, 0]
B_cov, S_cov, _, _ = var_ols(qdata, p=P_VAR, dummies=D)
fc_cov_full = var_forecast(B_cov, qdata, h=H, p=P_VAR, current=current, n_dummies=D.shape[1])

print("\n=== VAR(2) — GDP point forecast ===")
print(pd.DataFrame({"raw (2020 in)": fc_raw, "COVID dummy": fc_cov_full[:, 0]},
                   index=labels).round(2))
print("\n=== VAR(2) — GDP forecast with bands (COVID dummy) ===")
print(var_bands(B_cov, S_cov, fc_cov_full, p=P_VAR, n=4, labels=labels).round(2))

# --- BVAR(4): density forecast (raw vs dummy); paths stored for plot 4 ---
paths_raw = bvar(qdata, p=P_BVAR, levels=lev, current=current)
paths_dum = bvar(qdata, p=P_BVAR, levels=lev, dummies=D, current=current)
print("\n=== BVAR(4) — GDP forecast, RAW (2020 included) ===")
print(gdp_fan(paths_raw, labels).round(2))
print("\n=== BVAR(4) — GDP forecast, COVID DUMMY (2020 absorbed) ===")
print(gdp_fan(paths_dum, labels).round(2))

# ============================================================
#  STEP 4 — PSEUDO-REAL-TIME OUT-OF-SAMPLE EVALUATION
# ============================================================
# Expanding window from EVAL_START: at each quarter, use ONLY data up to that
# point, forecast h=1..4, compare with the realized value.
# DFM parameters are estimated once (full sample) and kept fixed; only the
# filter is re-run per window (standard shortcut). The AR(1) benchmark gets
# the COVID dummies too (as macro4.m) so the race is fair.

def bvar_mean(df, p=4, levels=None, dummies=None, **lam):
    """Minnesota BVAR, analytic posterior mean (= Gibbs median for point forecasts).
       Used inside the evaluation loop: no simulation needed for point accuracy."""
    lamd = {**LAM, **lam}
    data = df.values; n = data.shape[1]
    if levels is None: levels = [False] * n
    d = 0 if dummies is None else dummies.shape[1]
    Y = data[p:]
    X = build_X(data, p, dummies); k = X.shape[1]
    B0, hdiag, s = minnesota_prior(data, p, k, n, lamd["lam1"], lamd["lam2"],
                                   lamd["lam3"], lamd["lam4"], levels, d)
    B = np.zeros((k, n))
    for i in range(n):
        b0 = B0[i * k:(i + 1) * k]
        Vinv = np.diag(1.0 / hdiag[i * k:(i + 1) * k])
        s2 = s[i] ** 2
        B[:, i] = np.linalg.solve(X.T @ X / s2 + Vinv, X.T @ Y[:, i] / s2 + Vinv @ b0)
    return B

def ar1_forecast(y, h=4, dummies=None):
    """AR(1) benchmark with COVID dummies (macro4.m); future dummies = 0."""
    Y = y[1:]
    Xc = [np.ones(len(y) - 1), y[:-1]]
    if dummies is not None:
        Xc += [dummies[1:, j] for j in range(dummies.shape[1])]
    X = np.column_stack(Xc)
    b = np.linalg.solve(X.T @ X, X.T @ Y)
    f = []; last = y[-1]
    for _ in range(h):
        last = b[0] + b[1] * last; f.append(last)
    return np.array(f)

def dm_test(e1, e2, h=1):
    """Diebold-Mariano (1995) on squared errors, Newey-West variance (lag h-1).
       DM > 0 -> model 2 more accurate; ~N(0,1) under equal accuracy."""
    d = e1 ** 2 - e2 ** 2
    T = len(d); dbar = d.mean(); dd = d - dbar
    var = np.sum(dd ** 2) / T
    for l in range(1, h):
        var += 2 * (1 - l / h) * np.sum(dd[l:] * dd[:-l]) / T
    dm = dbar / np.sqrt(var / T)
    return dm, 2 * (1 - norm.cdf(abs(dm)))

# ---- 4a: expanding-window race, h=1..4 ----
print("\nSTEP 4 — out-of-sample evaluation (expanding window)...")
gdp_q = qdata["gdp"]
start = qdata.index.get_loc(pd.Timestamp(EVAL_START))
rows = []
for t0 in range(start, len(qdata) - 1):
    train = qdata.iloc[:t0 + 1]
    Dtr = D[:t0 + 1]; keep = Dtr.sum(axis=0) > 0      # dummies only once 2020 is in the window
    Dtr2 = Dtr[:, keep] if keep.any() else None
    nd = 0 if Dtr2 is None else Dtr2.shape[1]
    Bv, _, _, _ = var_ols(train, p=P_VAR, dummies=Dtr2)
    f_var = var_forecast(Bv, train, h=H, p=P_VAR, n_dummies=nd)[1:, 0]
    Bb = bvar_mean(train, p=P_BVAR, levels=lev, dummies=Dtr2)
    f_bvar = var_forecast(Bb, train, h=H, p=P_BVAR, n_dummies=nd)[1:, 0]
    f_ar = ar1_forecast(train["gdp"].values, h=H, dummies=Dtr2)
    f_rw = np.repeat(train["gdp"].iloc[-1], H)
    for h in range(1, H + 1):
        if t0 + h < len(qdata):
            rows.append(dict(origin=qdata.index[t0], target=qdata.index[t0 + h], h=h,
                             real=gdp_q.iloc[t0 + h], VAR=f_var[h - 1], BVAR=f_bvar[h - 1],
                             AR1=f_ar[h - 1], RW=f_rw[h - 1]))
res_oos = pd.DataFrame(rows)

def rmsfe_table(df):
    out = {m: [np.sqrt(((df[df.h == h][m] - df[df.h == h].real) ** 2).mean())
               for h in range(1, H + 1)] for m in ["AR1", "RW", "VAR", "BVAR"]}
    tab = pd.DataFrame(out, index=[f"h={h}" for h in range(1, H + 1)]).T
    return tab, tab / tab.loc["AR1"]

mask_ex = (res_oos.target < "2020-01") | (res_oos.target > "2020-12")
tab_all, rel_all = rmsfe_table(res_oos)
tab_ex,  rel_ex  = rmsfe_table(res_oos[mask_ex])
print("\n=== RMSFE relative to AR(1), FULL eval sample ===");  print(rel_all.round(3))
print("\n=== RMSFE relative to AR(1), EXCLUDING 2020 ===");    print(rel_ex.round(3))

print("\n=== Diebold-Mariano: VAR vs BVAR (excl. 2020) ===")
for h in range(1, H + 1):
    sub = res_oos[mask_ex & (res_oos.h == h)]
    dm, p = dm_test((sub.VAR - sub.real).values, (sub.BVAR - sub.real).values, h=h)
    print(f"  h={h}: DM={dm:+.2f} p={p:.3f}  -> better: {'BVAR' if dm > 0 else 'VAR'}")
print("=== Diebold-Mariano: BVAR vs AR(1) (excl. 2020) ===")
for h in range(1, H + 1):
    sub = res_oos[mask_ex & (res_oos.h == h)]
    dm, p = dm_test((sub.BVAR - sub.real).values, (sub.AR1 - sub.real).values, h=h)
    print(f"  h={h}: DM={dm:+.2f} p={p:.3f}  -> better: {'AR1' if dm > 0 else 'BVAR'}")

# ---- 4b: pseudo-real-time DFM nowcast (h=0) vs AR(1) ----
# At each quarter end: cut the panel, hide the target GDP, re-standardize on
# the window only (no look-ahead), re-run the filter, read the nowcast.
rng_nc = np.random.default_rng(1)
rows_nc = []
for cut in [d for d in panel.index if d.month in (3, 6, 9, 12) and d >= pd.Timestamp(EVAL_START)]:
    i = panel.index.get_loc(cut)
    sub = panel.iloc[:i + 1].copy()
    real = sub.loc[cut, "gdp"]
    if np.isnan(real): continue
    sub.loc[cut, "gdp"] = np.nan
    mu_w, sd_w = sub.mean(), sub.std()
    sz = (sub - mu_w) / sd_w
    iw = (~sz.isna()).astype(float).values
    yw = sz.values.copy()
    yw[np.isnan(yw)] = rng_nc.standard_normal(int(np.isnan(yw).sum()))
    _, fw = kalman_filter(theta_hat, yw, iw)
    nc_val = (fw[-1, 0:5] @ (theta_hat[8] * W_AGG)) * sd_w["gdp"] + mu_w["gdp"]
    g = sub["gdp"].dropna().values
    Xa = np.column_stack([np.ones(len(g) - 1), g[:-1]])
    b = np.linalg.solve(Xa.T @ Xa, Xa.T @ g[1:])
    rows_nc.append(dict(q=cut, nowcast=nc_val, ar1=b[0] + b[1] * g[-1], real=real))
r_nc = pd.DataFrame(rows_nc).set_index("q")

print("\n=== DFM nowcast (h=0) vs AR(1), pseudo-real-time ===")
m20 = (r_nc.index < "2020-01") | (r_nc.index > "2020-12")
for lbl, m in [("full", r_nc.index == r_nc.index), ("excl 2020", m20)]:
    s = r_nc[m]
    rn = np.sqrt(((s.nowcast - s.real) ** 2).mean())
    ra = np.sqrt(((s.ar1 - s.real) ** 2).mean())
    print(f"  {lbl:10s} n={len(s)}  RMSFE nowcast={rn:.3f}  AR1={ra:.3f}  ratio={rn/ra:.3f}")

# ============================================================
#  STEP 5 — THE FIVE DELIVERABLE PLOTS
# ============================================================
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "legend.frameon": False,
})
BLUE, RED, GRAY = "#2166ac", "#b2182b", "#666666"
Path("plots").mkdir(exist_ok=True)

# ---- Plot 1: intra-quarter nowcast evolution (the assignment's showcase) ----
def realtime_nowcast(target_end, vintage):
    """Nowcast of the quarter ending at target_end, seen at the vintage month:
       each series revealed only up to vintage minus its publication lag."""
    last = max(target_end, vintage)
    sub = panel.loc[:last].copy()
    for c in cols:
        avail = vintage - pd.DateOffset(months=PUB_LAGS[c])
        sub.loc[sub.index > avail, c] = np.nan
    mu_w, sd_w = sub.mean(), sub.std()
    sz = (sub - mu_w) / sd_w
    iw = (~sz.isna()).astype(float).values
    yw = sz.values.copy()
    rng = np.random.default_rng(0)
    yw[np.isnan(yw)] = rng.standard_normal(int(np.isnan(yw).sum()))
    _, fw = kalman_filter(theta_hat, yw, iw)
    i_t = sub.index.get_loc(target_end)
    return (fw[i_t, 0:5] @ (theta_hat[8] * W_AGG)) * sd_w["gdp"] + mu_w["gdp"]

targets = {"2011Q4 — recession": "2011-12-01",
           "2017Q2 — calm":      "2017-06-01",
           "2020Q2 — COVID":     "2020-06-01"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (qname, qend) in zip(axes, targets.items()):
    qend = pd.Timestamp(qend)
    vint = pd.date_range(qend - pd.DateOffset(months=5),
                         qend + pd.DateOffset(months=2), freq="MS")
    ncs = [realtime_nowcast(qend, v) for v in vint]
    final = gdp_real.get(qend, np.nan)
    ax.plot(vint, ncs, "o-", color=BLUE, lw=2.2, ms=6, zorder=3, label="nowcast")
    ax.axhline(final, color=RED, ls="--", lw=1.8, label=f"final GDP: {final:+.2f}%")
    ax.axvline(qend, color=GRAY, lw=1, ls=":")
    ax.annotate(f"{ncs[-1]:+.2f}", xy=(vint[-1], ncs[-1]), xytext=(6, 6),
                textcoords="offset points", fontsize=10, color=BLUE, fontweight="bold")
    ax.set_title(qname)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(loc="best")
axes[0].set_ylabel("GDP quarterly growth (%)")
fig.suptitle("The nowcast converges to the truth as monthly data arrive",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig("plots/p1_intraquarter.png", dpi=150, bbox_inches="tight")
plt.close(); print("plot 1 saved")

# ---- Plot 2: news decomposition = weight x surprise (§4.6) ----
def kalman_news(theta, yv_, mask_):
    """Per-period contribution of each series to the GDP-nowcast revision."""
    R, Hm, F, Q = matrices(theta)
    m = F.shape[0]; T, n = yv_.shape
    beta = np.zeros((m, 1)); P = np.eye(m); news = np.zeros((T, n))
    for t in range(T):
        idx = mask_[t]
        Hit = idx[:, None] * Hm; Rit = np.diag(1.0 - idx)
        beta10 = F @ beta; P10 = F @ P @ F.T + Q
        n10 = yv_[t].reshape(-1, 1) - Hit @ beta10
        F10 = Hit @ P10 @ Hit.T + Rit; F10 = 0.5 * (F10 + F10.T)
        F10i = np.linalg.inv(F10)
        K = P10 @ Hit.T @ F10i
        news[t] = (Hm[6:7, :] @ K).ravel() * n10.ravel()   # GDP-row weight x surprise
        beta = beta10 + K @ n10; P = P10 - K @ Hit @ P10
    return news

news = kalman_news(theta_hat, yv, obs_mask) * sd_p["gdp"]
nice = {"ip": "Industrial production", "retail": "Retail sales", "exports": "Exports",
        "unemp": "Unemployment", "bci": "Business confidence",
        "cci": "Consumer confidence", "gdp": "GDP release"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, (pick, title) in zip(axes, [("2020-04-01", "April 2020 — the crash arrives"),
                                    ("2017-05-01", "May 2017 — a calm month")]):
    i = dates.get_loc(pd.Timestamp(pick))
    vals = pd.Series(news[i], index=[nice[c] for c in cols]).sort_values()
    ax.barh(vals.index, vals.values, color=[RED if v < 0 else BLUE for v in vals], alpha=.9)
    ax.axvline(0, color="black", lw=1)
    lim = max(abs(vals.min()), abs(vals.max()))
    ax.set_xlim(-lim * 1.25, lim * 1.25)
    for y, v in enumerate(vals.values):
        ax.annotate(f"{v:+.2f}", xy=(v, y), xytext=(5 if v >= 0 else -5, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_title(f"{title}\n(total revision: {vals.sum():+.2f} pp)")
    ax.set_xlabel("contribution to nowcast revision (pp)")
fig.suptitle("News decomposition: weight × surprise, release by release",
             fontsize=14, fontweight="bold", y=1.04)
plt.tight_layout(); plt.savefig("plots/p2_news.png", dpi=150, bbox_inches="tight")
plt.close(); print("plot 2 saved")

# ---- Plot 3: common factor vs GDP growth, recessions shaded ----
fig, ax = plt.subplots(figsize=(12, 4.6))
has_gdp = ~gdp_real.isna()
gdp_std = (gdp_real - gdp_real.mean()) / gdp_real.std()
fac_std = (factor - factor.mean()) / factor.std()
for a, b, lbl in [("2008-04", "2009-06", "recessions"),
                  ("2011-07", "2013-03", None), ("2020-01", "2020-09", None)]:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d9d9d9", alpha=.6, label=lbl)
ax.plot(dates, fac_std, color=BLUE, lw=1.6, label="DFM common factor")
ax.plot(dates[has_gdp], gdp_std[has_gdp], "o", color="black", ms=4, alpha=.75,
        label="GDP growth (quarterly)")
ax.axhline(0, color=GRAY, lw=.8)
ax.set_ylim(-6, 6); ax.set_ylabel("standard deviations")
ax.set_title("The common factor tracks the Italian business cycle")
ax.legend(loc="lower left", ncol=3)
ax.xaxis.set_major_locator(mdates.YearLocator(4))
plt.tight_layout(); plt.savefig("plots/p3_factor.png", dpi=150, bbox_inches="tight")
plt.close(); print("plot 3 saved")

# ---- Plot 4: VAR vs BVAR forecasts, anchored on the nowcast, 80% bands ----
band_var = var_bands(B_cov, S_cov, fc_cov_full, p=P_VAR, n=4, labels=labels)
b10, b50, b90 = np.percentile(paths_dum[:, :, 0], [10, 50, 90], axis=0)

fig, ax = plt.subplots(figsize=(11.5, 5))
hist_gdp = qdata["gdp"].loc["2023-01":]
ax.plot(hist_gdp.index, hist_gdp.values, "o-", color="black", ms=5, lw=1.6,
        label="Realized GDP growth")
ax.plot(future_dates, fc_cov_full[:, 0], "s-", color=BLUE, lw=2, ms=6, label="VAR(2)")
ax.fill_between(future_dates, band_var["p10"], band_var["p90"], color=BLUE, alpha=.15,
                label="VAR 80% band")
ax.plot(future_dates, b50, "^-", color=RED, lw=2, ms=6, label="BVAR(4) Minnesota")
ax.fill_between(future_dates, b10, b90, color=RED, alpha=.12, label="BVAR 80% band")
ax.axhline(0, color=GRAY, lw=.8)
ax.axvline(qdata.index[-1] + pd.DateOffset(days=45), color=GRAY, ls=":", lw=1.2)
ax.annotate(f"nowcast {nowcast_now:+.2f}", xy=(future_dates[0], nowcast_now),
            xytext=(-75, 14), textcoords="offset points", fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GRAY))
ax.set_title("GDP growth: VAR vs BVAR forecasts, anchored on the DFM nowcast "
             "(80% bands, COVID dummies)")
ax.set_ylabel("GDP quarterly growth (%)")
ax.legend(loc="lower left", ncol=2)
plt.tight_layout(); plt.savefig("plots/p4_var_bvar.png", dpi=150, bbox_inches="tight")
plt.close(); print("plot 4 saved")

# ---- Plot 5: relative RMSFE bars (the Step 4 verdict in one picture) ----
fig, ax = plt.subplots(figsize=(10, 4.8))
xpos = np.arange(H); wd = 0.26
for i, (mn, color) in enumerate([("RW", "#999999"), ("VAR", BLUE), ("BVAR", RED)]):
    vals = rel_ex.loc[mn].values
    ax.bar(xpos + (i - 1) * wd, np.minimum(vals, 2.0), wd, label=mn, color=color, alpha=.9)
    for x, v in zip(xpos + (i - 1) * wd, vals):
        ax.annotate(f"{v:.2f}" if v <= 2 else f"{v:.1f}↑", xy=(x, min(v, 2.0)),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
ax.axhline(1.0, color="black", lw=1.4, ls="--")
ax.annotate("AR(1) benchmark = 1   (below the line = better than AR(1))",
            xy=(0.02, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 6),
            textcoords="offset points", fontsize=10)
ax.set_xticks(xpos)
ax.set_xticklabels([f"{h} quarter{'s' if h > 1 else ''}" for h in range(1, H + 1)])
ax.set_xlabel("forecast horizon"); ax.set_ylabel("RMSFE relative to AR(1)")
ax.set_ylim(0, 2.15)
ax.set_title("Out-of-sample forecast accuracy relative to AR(1), excluding 2020")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig("plots/p5_rmsfe.png", dpi=150, bbox_inches="tight")
plt.close(); print("plot 5 saved")

# ============================================================
#  STEP 6 — SAVED OUTPUTS & FINAL SUMMARY
# ============================================================
OUT = Path("data/output"); OUT.mkdir(parents=True, exist_ok=True)

forecast_table = pd.DataFrame({
    "VAR":      fc_cov_full[:, 0],
    "BVAR":     b50,
    "BVAR_p10": b10,
    "BVAR_p90": b90,
}, index=labels).round(3)

nowcast_table.round(3).to_csv(OUT / "nowcast_by_quarter.csv")
forecast_table.to_csv(OUT / "forecast_var_bvar.csv")
res_oos.round(4).to_csv(OUT / "oos_forecasts.csv", index=False)
rel_all.round(3).to_csv(OUT / "rmsfe_relative_full.csv")
rel_ex.round(3).to_csv(OUT / "rmsfe_relative_ex2020.csv")
r_nc.round(3).to_csv(OUT / "nowcast_oos_eval.csv")
np.save(OUT / "theta_hat.npy", theta_hat)

# nowcast accuracy ratios (vs AR(1), pseudo-real-time)
ratio_full = (np.sqrt(((r_nc.nowcast - r_nc.real) ** 2).mean())
              / np.sqrt(((r_nc.ar1 - r_nc.real) ** 2).mean()))
ratio_ex = (np.sqrt(((r_nc[m20].nowcast - r_nc[m20].real) ** 2).mean())
            / np.sqrt(((r_nc[m20].ar1 - r_nc[m20].real) ** 2).mean()))
# at how many horizons does the BVAR beat the VAR? (DM sign)
dm_wins = sum(dm_test((res_oos[mask_ex & (res_oos.h == h)].VAR
                       - res_oos[mask_ex & (res_oos.h == h)].real).values,
                      (res_oos[mask_ex & (res_oos.h == h)].BVAR
                       - res_oos[mask_ex & (res_oos.h == h)].real).values,
                      h=h)[0] > 0 for h in range(1, H + 1))

L = "=" * 64
print(f"""
{L}
  ITALIAN GDP — NOWCASTING & FORECASTING · FINAL SUMMARY
{L}
  Data      : 9 series, {dates[0].year}-{dates[-1].year} · monthly DFM panel + quarterly VAR set
  Models    : DFM (1 factor, mixed-freq, Kalman/MLE) · VAR({P_VAR}) · BVAR({P_BVAR}, Minnesota)

  NOWCAST — current quarter
    {cur_label}: {nowcast_now:+.2f}%   (GDP not yet published)
    accuracy vs AR(1), pseudo-real-time {pd.Timestamp(EVAL_START).year}-{dates[-1].year}:
      full sample : {ratio_full:.2f}   excl. 2020 : {ratio_ex:.2f}   (<1 = beats benchmark)

  FORECAST — next quarters (anchored on the nowcast, COVID dummies)""")
for q in labels[1:]:
    r = forecast_table.loc[q]
    print(f"    {q}:  VAR {r['VAR']:+.2f}   BVAR {r['BVAR']:+.2f}  "
          f"[{r['BVAR_p10']:+.2f}, {r['BVAR_p90']:+.2f}]")
print(f"""
  MODEL RANKING — out-of-sample, excl. 2020 (RMSFE vs AR(1))
    h=1..4  VAR : {'  '.join(f"{v:.2f}" for v in rel_ex.loc['VAR'])}
    h=1..4  BVAR: {'  '.join(f"{v:.2f}" for v in rel_ex.loc['BVAR'])}
    BVAR more accurate than VAR at {dm_wins}/{H} horizons (DM sign; not significant at 5%)

  Outputs   : data/output/*.csv · plots/p1-p5.png
{L}""")