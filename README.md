# Nowcasting & Forecasting Italian GDP
### DFM/Kalman nowcast + VAR/BVAR forecast 

Author: Vassily Dal Bo

---

## What this project does

1. **Nowcast** of current-quarter Italian GDP growth with a **mixed-frequency dynamic
   factor model** (1 common factor, GDP enters the monthly model through the triangular
   weights 1/3, 2/3, 1, 2/3, 1/3), estimated by **MLE via the Kalman filter**, which
   handles the ragged edge and missing data natively (the Ht/Rt trick).
2. **Forecast** of the next 4 quarters with a **VAR(2)** (OLS) and a **BVAR(4)**
   (Minnesota prior, Gibbs sampling), anchored on the DFM nowcast for the current
   quarter.
3. **Pseudo-real-time out-of-sample evaluation** (expanding window, 2012→today):
   nowcasts built only with the data published at each date (publication lags) and
   a DFM re-estimated every year on that information set; forecasts h=1..4 for
   AR(1), random walk, VAR, BVAR and the BVAR anchored on the nowcast;
   Diebold-Mariano tests with the Harvey-Leybourne-Newbold correction, with and
   without 2020.

**COVID** is handled consistently in both blocks: dummies for 2020Q1-Q3 in the
VAR/BVAR (and in the AR(1) benchmark), and March-September 2020 treated as missing
in the DFM likelihood and in the standardization. The DFM still filters those
months, so it produces a nowcast for 2020, but the 2020 swings no longer drive the
parameters (with them, the GDP loading estimated after 2020 jumped from 0.26 to 0.78).

## How to run

```bash
pip install -r requirements.txt
python main.py          # full pipeline, ~4 minutes
```

Outputs: `plots/p1-p5.png` (the five deliverable figures) plus
`plots/p6_nowcast_vintages.png` (nowcast accuracy within the quarter),
`data/output/*.csv` (nowcast table, forecasts, out-of-sample results, real-time
parameter estimates), and a final summary printed to the console.

## Data (raw CSVs in `RAW/`)

| Series | Source | Code / dataset | Transform |
|---|---|---|---|
| Real GDP (quarterly, SA) | FRED | CLVMNACSCAB1GQIT | Δlog ×100 |
| Exports (monthly, SA) | FRED | XTEXVA01ITM667S | Δlog ×100 |
| Unemployment rate (monthly, SA) | FRED | LRHUTTTTITM156S | Δ |
| HICP (monthly, NSA) | FRED | CP0000ITM086NEST | Δlog ×100, seasonally adj. in code |
| 3-month rate (monthly) | FRED | IR3TIB01EZM156N | Δ |
| Industrial production (monthly, SCA, 2021=100) | Eurostat | sts_inpr_m (B-D, I21) | Δlog ×100 |
| Retail sales volume (monthly, SCA, 2021=100) | Eurostat | sts_trtu_m (G47, I21) | Δlog ×100 |
| Business confidence (monthly, SA) | Eurostat | ei_bsin_m_r2 (BS-ICI) | level |
| Consumer confidence (monthly, SA) | Eurostat | ei_bsco_m (BS-CSMCI) | level |

Estimation sample: 2000M1 onward. Soft indicators load on a **12-month sum of the
factor** (surveys track year-on-year dynamics). Confidence and production/retail come
from Eurostat because the corresponding FRED (OECD) series were discontinued in 2024.

## Key results (excluding 2020 unless noted)

- **Nowcast 2026Q2: +0.12%.** BVAR path afterwards: +0.32, +0.12, +0.04, +0.05
  (80% band roughly ±0.8 pp).
- **DFM nowcast vs AR(1)**, pseudo-real-time, relative RMSFE by vintage:
  0.88 (month 1), 1.00 (month 2), 0.93 (month 3), **0.85** (one month after the
  quarter, just before the GDP release). The error falls from 0.68 to 0.51 pp as
  the monthly data arrive. Gains are modest and not significant (DM-HLN); including
  2020 the ratio goes down to 0.55, because the AR(1) cannot see the crash coming.
- **BVAR vs VAR**: the BVAR has lower RMSFE at **all** horizons h=1..4
  (0.97, 0.82, 0.80, 0.82 vs 1.19, 0.94, 0.88, 0.96, relative to AR(1)); the DM-HLN
  test is significant at 5% only at h=2 (p = 0.05), with ~50 observations.
- **BVAR anchored on the nowcast** (the pipeline used for the forecasts): best
  model at h=2 and h=3 (0.79, 0.77); at h=1 it is equivalent to the AR(1) (0.99).
- **Random walk**: best model at h=1 (0.83: growth is persistent from one quarter to
  the next) but dominated from h=2 on (2.3-3.7) because quarterly growth is
  mean-reverting.

## Code structure (single script, `main.py`)

| Section | Content |
|---|---|
| CONFIG | every modelling choice in one place |
| HELPERS | shared lag-matrix builder and Minnesota prior |
| STEP 1 | data loading, transformations, common monthly calendar |
| STEP 2 | state-space matrices, Kalman filter (= likelihood), MLE, nowcast |
| STEP 3 | VAR(2), BVAR(4) Gibbs, COVID dummies, forecasts with bands |
| STEP 4 | real-time DFM re-estimation, nowcast by vintage, expanding-window forecast race, RMSFE ratios, Diebold-Mariano (HLN) |
| STEP 5 | the five deliverable plots + nowcast accuracy by vintage |
| STEP 6 | saved outputs + final summary |

Known limitations (to be discussed in the final report): final-vintage data (no
real-time vintages, so GDP revisions are ignored); calendar-based publication lags
(fixed lag per series, no exact release dates); in the evaluation the anchored BVAR
uses only the GDP nowcast, not the partial monthly data on inflation and
unemployment; the optimizer stops at `success=False` (flat likelihood near the
optimum; estimates stable across the yearly re-estimations, see
`data/output/theta_realtime.csv`).
