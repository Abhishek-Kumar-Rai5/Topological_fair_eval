# Implementation notes

These are choices the preregistration leaves open. None of them changes a prereg rule. All were fixed **before any detector was run on real data**, and have not been changed after seeing results. If one changes after results exist, it goes in `deviations_log.md` instead.

## Pipeline (step 8, `src/`)

| # | Where | Prereg text | Our choice |
|---|---|---|---|
| 1 | `threshold.py` | 1% false-alarm rate on validation windows | At most ⌊1% × n⌋ validation windows score strictly above the threshold, so the realised rate can be slightly below 1% (SMD: about 470–570 validation windows per machine, so at most 4–5 alarms). Ties at the threshold do not alarm |
| 2 | `metrics.py` | false alarms per day; consecutive alarms count as one | A run of consecutive false-alarm windows (alarm on a NORMAL window) is one event. Any other window (no alarm, positive, excluded) ends the run. Days = test length × 1 minute / 1440 |
| 3 | `splits.py` | last 20% of training = validation | `round(0.2 × n_train)` rows. Windows are cut separately in each part, so no window spans the boundary |
| 4 | `metrics.py` | PR-AUC, ROC-AUC, recall | Computed per machine on its own test windows, excluding label −1, then averaged over seeds and bootstrapped over machines |
| 5 | §8 inconclusive rule | "cells with fewer than 20 positive or fewer than 20 normal test windows" | Read as one comparison cell over all 28 machines, not one machine. machine-1-5 (16 positives) and machine-2-8 (17) stay in. **Awaiting Anirudh's confirmation** |
| 6 | `detector.py` | Isolation Forest, fixed defaults | `IsolationForest(n_estimators=300, max_samples="auto", random_state=seed)` after a `StandardScaler` fitted on training features. Score = −`score_samples` (higher = more anomalous) |

## F1 and F2 features (step 9, `src/features_simple.py`)

| # | Feature | Prereg text | Our choice |
|---|---|---|---|
| 7 | F1 unique count | per channel unique count | Exact count of distinct values in the window |
| 8 | F2 correlation summary | mean, min, max, std of off-diagonal values; top-eigenvalue share | Pearson correlation between channels within a window. Constant channels (max = min) are left out. With fewer than 2 varying channels, all 5 values are 0. Top-eigenvalue share = largest eigenvalue / sum of eigenvalues |
| 9 | F2 periodicity | FFT peak share; maximum autocorrelation at lags 2–30 | **Per channel**, as in F1 (2 × 38 = 76 features). FFT: remove the mean, drop frequency 0, peak power / total power. ACF: standard estimator Σ xₜxₜ₊ₖ / Σ xₜ² on the mean-removed channel, maximum over k = 2..30. Constant channels get 0 |
| 10 | F2 PCA error | PCA reconstruction error, fitted on training data | Scaler and PCA fitted on all time points of the training windows (38-dim points). Number of components: enough for **95% of training variance** (SMD: 8–20 per machine). Window feature = mean squared reconstruction error over its 60 points (1 feature) |
| 11 | constant channels | — | "Constant" means max − min = 0 (exact). `std > 0` is not used, because a constant 0.1 has std of about 1e-17 from rounding; this bug was found and fixed during the real-data check |

Feature sizes on SMD: F1 = 152, F2 = 82 (5 + 76 + 1), F1+F2 = 234.
