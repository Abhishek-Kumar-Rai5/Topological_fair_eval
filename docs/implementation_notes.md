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

## F3 topology features (step 10, `src/features_topo.py`) — **confirmed by Anirudh, 2026-11-19**

Prereg §4: "Delay embedding (m = 3, τ = 1). Vietoris–Rips persistence via `ripser`, H0 and H1. Summarised by: number of bars; total, max, mean and std of lifetimes; persistent entropy; Betti curve at 10 fixed multiples of the median pairwise distance. 32 features, no tuning." All four choices below were fixed before any detector ran, and are logged as clarifications in `deviations_log.md`.

| # | Point | Our choice | Why |
|---|---|---|---|
| 12 | What is embedded | The **whole 38-channel window**: point t = [xₜ, xₜ₊₁, xₜ₊₂] with xₜ ∈ ℝ³⁸, so 58 points in ℝ¹¹⁴, one point cloud and one diagram pair per window | "32 features" = 2 dims × 16 summaries, which only works with one diagram per window. Per-channel embedding would give 38 × 32 = 1,216 features |
| 13 | Scaling before distances | **Z-score each channel with the machine's training mean and std** (all time points of the training windows), std floored at **0.01**. Fitted on training data only | Prereg: every fitted transform uses training data only. The floor stops near-constant channels from blowing up: on SMD, 192 of 1,064 machine-channels have training std exactly 0 and 337 have std below 0.01. The floor value is 1% of the [0, 1] range; it is a fixed constant, not tuned |
| 14 | Betti curve scales | **0.2, 0.4, …, 2.0 × the training median pairwise distance** (the multiples from prereg v2). The median is computed **once per machine**, pooled over all pairwise distances of all training windows (in z-scored space), and is the same for every window. A bar is alive at s if birth ≤ s < death | A per-window scale would divide away changes in overall size (volume) and rig the volume-vs-shape test (E5) |
| 15 | Infinite H0 bar | **Dropped everywhere**, including the Betti curve. All summaries use finite bars only. Zero-length bars are not reported by ripser. Std uses ddof = 0; entropy uses natural log; an empty diagram gives 0s | Every window has exactly one infinite H0 bar, so it carries no information |

Feature sizes: F3 = 32; F1+F2+F3 = 266. Real-data check (machine-1-1 and machine-3-11): all values finite, about 1.2 ms per window, fit about 0.4 s per machine, H1 bars per window median 22 and 29. The training median distance is 3.9 and 5.6.

**Std floor 0.01 (F3 choice 13): accepted by Anirudh, 2026-11-19.** It stays logged in `deviations_log.md`.

## F5 Fluent Bit-style features (step 11, `src/features_fluentbit.py`) — exploratory

Prereg §4: "Replicates the `tda` processor: log1p transform, embedding 3 × 1, a quantile radius scan that keeps the scale with maximum β1, then outputs β0, β1, β2 (3 features). **Exploratory**". Logic copied from `tda.c` at `25ea7e1` (see `docs/external_versions.md`), defaults kept (`embed_dim` 3, `embed_delay` 1, `min_points` 10, `threshold` 0 = automatic scan). Details in the module docstring. Only choices the C code does not settle are listed here:

| # | Point | Our choice | Why |
|---|---|---|---|
| 16 | Input to the transform | **F5 replicates `tda.c` L561–569 literally: every metric is first-differenced** (`rate = (x_t − x_(t−1)) / dt`), then `sign(rate) · log1p(abs(rate))`. A 60-sample window gives 59 vectors → 57 embedded points. An earlier "gauges, no differencing" clarification was withdrawn before any results. Decided by Anirudh, 2026-11-19 | F5 exists to test the tool as shipped, so faithfulness to the code comes first. The code differences counters, gauges and untyped alike |
| 17 | `dt` | **1 sample** (one time unit) | SMD has no timestamps. The 1-minute interval (a secondary-source value) would only rescale the rates; dt = 1 keeps F5 independent of it |
| 18 | Feature groups | **One group per channel, D = 38** (the processor sums metrics into (namespace, subsystem) groups, about 8 in its docs). Confirmed by Anirudh, 2026-11-19 | SMD has no namespaces. Distances are over 38 × 3 = 114 dimensions, as in F3 |
| 19 | What is counted as beta_k | **Finite bars with persistence ≥ 1e-3** per dimension, not infinite bars (the code's real behaviour; confirmed by Anirudh, 2026-11-19) | That is what the wrapper code does; its comment says the opposite. See `docs/code_verification.md` §4. The 1e-3 is absolute, and SMD values lie in [0, 1] |
| 20 | Ripser | `ripser.py` (`ripser`, modulus 2) with `thresh` = the quantile distance and `maxdim` = 2; distances in float32, like the C code | Same algorithm as the bundled ripser 1.2.1. Not run against the C binary; checked on a hand-worked example (unit square) in `tests/test_features_fluentbit.py` |
| 21 | Scaling | None (rates of the provider-scaled SMD values). Nothing is fitted | The processor fits nothing |

Cost: about 39 ms per window (machine-1-1); about 1.5 hours single-threaded for all SMD windows. No random seed is involved.

## CTU-13 (E4)

- **Behaviour types: the label strings are not used** (they are traffic types: DNS, SPAM, CC1, HTTP-Ad, Binary-Download, not behaviours). Each scenario gets the type of its **README-listed** behaviours, and **Mixed** if they disagree. This is the prereg §2 fallback, so it is **not a deviation**. Decided by Anirudh, 2026-11-19.
- **Infected hosts** = the IPs the README lists as `Label: Botnet`. Other IPs that send From-Botnet flows (scenarios 3 and 12) are not infected hosts. Confirmed by Anirudh, 2026-11-19.
- E4 limitation (data fact, no rule change): see `deviations_log.md`.
