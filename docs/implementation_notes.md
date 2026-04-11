# Implementation notes

These are choices the preregistration leaves open. None of them changes a prereg rule. All were fixed **before any detector was run on real data**, and have not been changed after seeing results. If one changes after results exist, it goes in `deviations_log.md` instead.

## Pipeline (step 8, `src/`)

| # | Where | Prereg text | Our choice |
|---|---|---|---|
| 1 | `threshold.py` | 1% false-alarm rate on validation windows | At most ⌊1% × n⌋ validation windows score strictly above the threshold, so the realised rate can be slightly below 1% (SMD: about 470–570 validation windows per machine, so at most 4–5 alarms). Ties at the threshold do not alarm |
| 2 | `metrics.py` | false alarms per day; consecutive alarms count as one | A run of consecutive false-alarm windows (alarm on a NORMAL window) is one event. Any other window (no alarm, positive, excluded) ends the run. Days = test length × 1 minute / 1440 |
| 3 | `splits.py` | last 20% of training = validation | `round(0.2 × n_train)` rows. Windows are cut separately in each part, so no window spans the boundary |
| 4 | `metrics.py` | PR-AUC, ROC-AUC, recall | Computed per machine on its own test windows, excluding label −1, then averaged over seeds and bootstrapped over machines |
| 5 | §8 inconclusive rule | "cells with fewer than 20 positive or fewer than 20 normal test windows" | **Confirmed by Anirudh; logged as a clarification in `deviations_log.md`.** Counts are pooled across all machines in the cell (dataset × anomaly type). PR-AUC is computed per machine and averaged over the machines with at least one positive window in the cell (others skipped), and the bootstrap resamples those machines. machine-1-5 (16 positives) and machine-2-8 (17) stay in |
| 6 | `detector.py` | Isolation Forest, fixed defaults | `IsolationForest(n_estimators=300, max_samples="auto", random_state=seed)` after a `StandardScaler` fitted on training features. Score = −`score_samples` (higher = more anomalous) |

## F1 and F2 features (step 9, `src/features_simple.py`)

| # | Feature | Prereg text | Our choice |
|---|---|---|---|
| 7 | F1 unique count | per channel unique count | Exact count of distinct values in the window |
| 8 | F2 correlation summary | mean, min, max, std of off-diagonal values; top-eigenvalue share | Pearson correlation between channels within a window. Constant channels (max = min) are left out. With fewer than 2 varying channels, all 5 values are 0. Top-eigenvalue share = largest eigenvalue / sum of eigenvalues |
| 9 | F2 periodicity | FFT peak share; maximum autocorrelation at lags 2–30 | **Per channel**, as in F1 (2 × 38 = 76 features). FFT: remove the mean, drop frequency 0, peak power / total power. ACF: standard estimator Σ xₜxₜ₊ₖ / Σ xₜ² on the mean-removed channel, maximum over k = 2..30. Constant channels get 0 |
| 10 | F2 PCA error | PCA reconstruction error, fitted on training data | Scaler and PCA fitted on all time points of the training windows (38-dim points). Number of components: enough for **95% of training variance** (SMD: 8–20 per machine); **confirmed by Anirudh and logged as a clarification in `deviations_log.md`**. Window feature = mean squared reconstruction error over its 60 points (1 feature) |
| 11 | constant channels | — | "Constant" means max − min = 0 (exact). `std > 0` is not used, because a constant 0.1 has std of about 1e-17 from rounding; this bug was found and fixed during the real-data check |

Feature sizes on SMD: F1 = 152, F2 = 82 (5 + 76 + 1), F1+F2 = 234.

## F3 topology features (step 10, `src/features_topo.py`) — **pending Anirudh's confirmation**

Prereg §4: "Delay embedding (m = 3, τ = 1). Vietoris–Rips persistence via `ripser`, H0 and H1. Summarised by: number of bars; total, max, mean and std of lifetimes; persistent entropy; Betti curve at 10 fixed multiples of the median pairwise distance. 32 features, no tuning."

| # | Point | Our choice | Why |
|---|---|---|---|
| 12 | What is embedded | The **whole 38-channel window**: point t = [xₜ, xₜ₊₁, xₜ₊₂] with xₜ ∈ ℝ³⁸, so 58 points in ℝ¹¹⁴, one point cloud and one diagram pair per window | "32 features" = 2 dims × 16 summaries, which only works with one diagram per window. Per-channel embedding would give 38 × 32 = 1,216 features |
| 13 | Scaling before distances | **None.** The provider-scaled SMD values (already in [0, 1]) are used as given | "No tuning", and nothing to fit. A training-fitted standardiser would blow up near-constant channels and let their noise dominate the distances |
| 14 | Betti curve scales | **0.1, 0.2, …, 1.0 × the window's own median pairwise distance**; a bar is alive at s if birth ≤ s < death | Prereg says "10 fixed multiples" but not which. Per-window median makes the curve scale-free; ≤ 1.0× because H0 bars die and H1 bars live mostly below the median distance |
| 15 | Bar bookkeeping | H0's one infinite bar is excluded from the bar count and lifetime statistics but counted in the Betti curve. Zero-length bars are not reported by ripser. Std uses ddof = 0; entropy uses natural log; empty diagram gives 0s | Every window has exactly one infinite H0 bar, so it carries no information for the statistics, but it is a real component for the Betti number |

Feature sizes: F3 = 32; F1+F2+F3 = 266. Real-data check: all 141,226 SMD windows finite, about 1.6 ms per window; H1 bars per window median 20 (range 0–58).
