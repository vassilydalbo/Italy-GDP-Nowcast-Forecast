# Nowcasting & Forecasting Italian GDP
### DFM/Kalman nowcast + VAR/BVAR forecast — project based on the BSE Summer School course (G. Pérez-Quirós)

Author: Vassily Dal Bo

---

## What this project does

1. **Nowcast** of current-quarter Italian GDP growth with a **mixed-frequency dynamic
   factor model** (1 common factor, GDP enters the monthly model through the triangular
   weights 1/3, 2/3, 1, 2/3, 1/3), estimated by **MLE via the Kalman filter**, which
   handles the ragged edge and missing data natively (the Ht/Rt trick).
2. **Forecast** of the next 4 quarters with a **VAR(2)** (OLS) and a **BVAR(4)**
   (Minnesota prior, Gibbs sampling), anchored on the DFM nowcast for the current
   quarter. COVID is treated explicitly with dummies (2020Q1-Q3).
3. **Pseudo-real-time out-of-sample evaluation** (expanding window, 2012→today):
   RMSFE relative to AR(1) and random walk, Diebold-Mariano tests, with and
   without 2020.

## How to run

```bash
pip install -r requirements.txt
python main.py          # full pipeline, ~3-4 minutes
```

Outputs: `plots/p1-p5.png` (the five deliverable figures), `data/output/*.csv`
(nowcast table, forecasts, out-of-sample results), and a final summary printed
to the console.

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

- **DFM nowcast vs AR(1)**: relative RMSFE **0.64** (0.30 including 2020) —
  the monthly information set beats the univariate benchmark by ~36%.
- **BVAR vs VAR**: the BVAR has lower RMSFE at **all** horizons h=1..4
  (e.g. 0.81 vs 0.91 at h=3, relative to AR(1)); the DM test favours the BVAR
  at every horizon but differences are not significant at 5% (~50 obs).
- **Random walk**: dominated everywhere — quarterly GDP growth is mean-reverting.

## Code structure (single script, `main.py`)

| Section | Content |
|---|---|
| CONFIG | every modelling choice in one place |
| HELPERS | shared lag-matrix builder and Minnesota prior |
| STEP 1 | data loading, transformations, common monthly calendar |
| STEP 2 | state-space matrices, Kalman filter (= likelihood), MLE, nowcast |
| STEP 3 | VAR(2), BVAR(4) Gibbs, COVID dummies, forecasts with bands |
| STEP 4 | expanding-window evaluation, RMSFE ratios, Diebold-Mariano |
| STEP 5 | the five deliverable plots |
| STEP 6 | saved outputs + final summary |

Known limitations (discussed in the report): final-vintage data (no real-time
vintages), simple calendar-based publication lags, DFM parameters estimated once
(filter-only updates in the evaluation loop), optimizer stops at `success=False`
(flat likelihood near the optimum; estimates stable across restarts).
