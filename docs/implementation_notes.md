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

## E5 injections (`src/injections.py`, `src/e5.py`, `scripts/summarize_e5.py`)

Prereg §3 and §7 fix the six types, magnitudes, duration (60), count (10 per type × magnitude × machine) and the rule "normal test segments only". The points below are what the prereg leaves open. All were fixed before any E5 run. Confirmed items are marked.

| # | Point | Our choice |
|---|---|---|
| 22 | Positive window (**confirmed by Anirudh, 2026-11-19**) | The §2 rule: a window is positive if an injected point lies in its **final 10 samples**. A window that overlaps an injection only earlier is **excluded**. A window that touches a real labelled anomaly anywhere is **excluded** (so positives are purely injected and normals purely normal). Normal = no injected and no real labelled point in the window |
| 23 | Channels for V1–V3 (**confirmed**) | Any channel with **training std ≥ 0.01** (same floor as F3), chosen per injection at random. σ is that channel's training std. Constant and near-constant channels are excluded, because "+2σ" of a std of 1e-5 would be invisible. All 38 channels qualify except those machine-channels (about a third of channels on SMD have std below 0.01) |
| 24 | Fewer than 10 fit (**confirmed**) | We place as many as fit and **report every machine × type with fewer than 10** in `E5_placement.csv` (columns: n_placed, reason). `summarize_e5.py` prints them. A type with no eligible channel on a machine gets 0 sites and the reason "no eligible channel" |
| 25 | V3 "variance ×c" (**confirmed: literal**) | The **variance** is multiplied by c: deviations from the segment mean are multiplied by √c. The segment mean is kept |
| 26 | Dominant lag | The highest **local maximum** of the training autocorrelation over lags 2–30 (the F2 lag range). Periodic if that ACF value is > 0.5. A drifting channel has a high but monotone ACF, so it has no peak and is not periodic |
| 27 | S1 factor | The period is **divided** by f: the segment is read f times faster (linear interpolation), so f = 2 halves the period |
| 28 | S2 | Shifts the **periodic** channel of a coupled pair (a phase shift needs a period); the partner is untouched. Shift = fraction × that channel's dominant lag, by linear interpolation of the earlier series. A pair needs at least one periodic channel |
| 29 | S3 | Replaces a block of round(q × 60) samples (q = magnitude) with the same block of a random 60-sample, label-free segment of the same channel, taken ≥ 60 samples away from the unmodified test series. The partner is untouched |
| 30 | V1, V2 | V1 adds +kσ to the whole segment of one channel. V2 adds +kσ to 3 distinct random points of one channel |
| 31 | Spacing | At least **120 samples between the end of one injection and the start of the next** (so at least 180 start to start). This satisfies both readings of "≥ 120 apart" |
| 32 | Never touch labelled anomalies | The 60-sample footprint, and the samples an injection reads from (S1 up to 3 × 60, S2 up to half a period back), contain no real labelled point |
| 33 | Eligibility data | Computed on the **first 80% of the training file** (the fit part), so validation data is not even used to choose channels. Pairs: Pearson |r| > 0.7 among non-constant channels |
| 34 | One site set per (machine, type) | Sites, channels and random draws depend on (machine, type) only, so the three magnitudes are paired. The detector seed does not change them |
| 35 | Detector and threshold | Fitted once per (machine, seed, feature set) on the unmodified training windows. Threshold at 1% on validation. They are identical for every injected series, because injections never touch training or validation |
| 36 | Speed shortcut | Only windows overlapping an injection are recomputed; other windows reuse the features of the unmodified test series. Exact, because every feature is computed per window (checked in `tests/test_e5.py`) |
| 37 | P2 Bonferroni | A volume type "helps" for P2 if the paired Δ is ≥ 0.02 **and** the lower end of a CI at level 1 − 0.05/3 (98.3%) is above 0. P2 is Supported only if all three volume types are "No added value" at the decision magnitude; otherwise Inconclusive |
| 38 | Result files | `by_machine/<m>.csv` (written last, marks the machine as done), `by_machine/<m>_placement.csv`, merged `E5_results.csv` and `E5_placement.csv`, then `E5_summary.csv`, `E5_verdicts.csv`, `E5_predictions.csv`, `E5_curves.png` |

Atomic writes (all result files): written to a temp file in the same folder, flushed, then renamed; the sidecar is written after the data file. `--resume` re-runs a machine unless its file is **valid**: non-empty, parses, has at least one row, and its sidecar exists, parses and records the same row count (`src/metadata.is_valid_result`).

## E2 re-tested methods (`src/e2_methods.py`, run with `scripts/run_experiment.py configs/e2_fair_retest.yaml --methods`)

Prereg §5: TADA and PHANTOM with default parameters; TAME with two changes only (our threshold, detector fixed in advance to Isolation Forest). The runner sets the threshold (1% false alarms on validation windows) for every method; the adapters never see labels. Results go to `results/E2m/` so the E2 file for our own pipelines is not touched. A method that raises is written to `E2m_failures.csv` (never hidden); the other methods still run. **These are my choices; please confirm or change them before the run.**

| # | Point | Our choice |
|---|---|---|
| 39 (**approved by Anirudh, 2026-11-20**) | Windows | All methods are scored on **our windows** (60 samples, step 10), our split (fit part 80%, validation last 20%) and our labels, so they are comparable with F1+F2 and F1+F2+F3. TAME's released protocol (window 200, step 200, all 28 machines pooled into one stream, validation 18%) is **not** used for E2; it belongs to E1. This is a third change to TAME beyond the two in §5 and needs your OK |
| 40 (**approved by Anirudh, 2026-11-20**) | TADA windows | `tdaad` defaults (window 100, step 5, `tda_max_dim` 1, 5 centres; `contamination` unused). Our window ending at e gets the score of the TADA window **ending at e** (start e − 100), so it sees only the past; the first four windows (e < 100) take TADA's first window. Anomaly score = −(its robust Mahalanobis score). Uses `_raw_score_samples` (per-window scores), not the public `score_samples`, which sums the scores of all windows covering a time step and would look 90 samples into the future |
| 41 | TADA seeds | `random_state` = our seed (affects the robust covariance fit). The k-means inside `tdaad` is hard-coded to seed 42 |
| 42 | PHANTOM distance | **`main.py` scores with the Wasserstein distance** (`compare(..., 2)`); `AnomalyScore.compare` defaults to bottleneck (method 0), but the script never uses the default. CLAUDE.md §4 says "bottleneck"; **that line should say Wasserstein (main.py) — please check and correct it**. We follow `main.py` |
| 43 (**approved by Anirudh, 2026-11-20**) | PHANTOM vector | Per channel: mean, median, min, max (PHANTOM's `mean_values`, `median_values`, `min_value`, `max_value`) → 152 numbers per window. The released config is for log records, so this part has to be chosen |
| 44 | PHANTOM order | Baseline = first 4 training windows (`BASELINE_LENGTH 4000 / WINDOW_LENGTH 1000`, probability 1). The rest of the fit windows are streamed (rolling baseline update only if score < 0.1, the released `ANOMALY_THRESHOLD`). Then the validation windows are scored (state continues), then the test windows are scored from a **copy** of that state. The test pass also rolls the baseline, using scores only. Deterministic: the 5 seeds are copies of one run |
| 45 | PHANTOM ripser | `ripser.ripser(D, distance_matrix=True)["dgms"][0]` as in `main.py`; persim ignores the infinite bar (warning suppressed) |
| 46 | TAME features | TAME's own `TDAPersistenceImageExtractor` imported from `external/TAME/utils/tda_features.py` (3×1 delay embedding, VR H0+H1, 40×40 persistence images, σ = 1), then `StandardScaler` + PCA(50) fitted on the training windows. Not copied: TAME has no licence. Not z-scored per channel (SMD loader does not) |
| 47 | TAME detectors | **IF is the one fixed in advance** (200 trees, as in TAME, seed = ours). GMM (4 components, full, reg 1e-4) and One-Class SVM (ν 0.01, γ 0.1) use TAME's settings and are reported separately (`TAME-GMM`, `TAME-OCSVM`). **Deep SVDD is not run** (needs torch, not installed); it is covered by E1. "TAME (best)" is never computed |
| 48 (**done 2026-11-20**: `.venv-tame`, Python 3.11, numpy 1.26.4, scikit-learn 1.3.2, giotto-tda 0.6.2, CPU torch 2.15.0, no conflicts) | **TAME needs giotto-tda, not installed in the main venv.** `pip install --dry-run giotto-tda` in the main venv would **downgrade numpy to 1.26.4 and scikit-learn to 1.3.2**, which would change the library versions of every other result. Use a **separate venv** for the TAME run (or install with `--no-deps` and check that it imports). Until then `--methods` records TAME failures in `E2m_failures.csv` and still runs TADA and PHANTOM. Tests use a stand-in feature function, and one test uses the real extractor when `gtda` is importable |

Runtime (toy 38-channel machine of 25,000 samples, one seed): PHANTOM about 2 s, TADA about 5 s. Real data will differ. TAME's cost cannot be measured until giotto-tda is installed.

## TAME runs (2026-11-20)

- **E2, TAME on our windows:** `scripts/run_experiment.py configs/e2_tame.yaml --methods` in `.venv-tame` (untracked-config copy of `configs/e2_fair_retest.yaml` with TAME only), results in `results/E2m_tame/`. TADA and PHANTOM: `configs/e2_tada_phantom.yaml`, `results/E2m/`. Runtimes of TAME-GMM and TAME-OCSVM are detector scoring only; the shared feature extraction is counted in TAME-IF.
- **E2/P3, fixed TAME on its own windows:** `scripts/run_tame_own_windows.py` (TAME's released SMD protocol: one pooled stream, window 200 / step 200, 18% validation, seeds 42, 0, 1, 2, 3; TAME's own feature code imported from `external/TAME`) with exactly the two §5 changes. Per-machine numbers use only the windows lying inside one machine's test span. giotto-tda was limited to 2 cores (TAME's own code uses all).
- **E1:** TAME's released `notebooks/SMD.ipynb`, run unchanged with `nbconvert` in a copy (`external/TAME_e1`, so the shipped `results/` stay), CPU limited to 4 cores. The notebook's plotting cell failed (`boxplot() got an unexpected keyword argument 'labels'`, matplotlib 3.11), after the results file was written; the results are taken from that file. OneClassSVM (0.798 vs 0.555 shipped) and DeepSVDD (0.587 vs 0.611) differ from the shipped file; Isolation Forest (0.801) and GMM (0.808) reproduce it.
