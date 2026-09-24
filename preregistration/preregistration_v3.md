# Pre-registration v3: When does topology help? (fair re-evaluation)

- **Written:** 2026-11-19, before any SMD or CTU-13 data was accessed.
- **Replaces:** v2 (2026-11-16). The change log is at the end.
- **Freezing:** this file is frozen by a public commit (tag `prereg-v3`).
- **Changes after the freeze:** any later change goes in the deviations log (§9).

## 1. Predictions

| ID | Prediction | Tested by |
|---|---|---|
| P1 | Topology adds detection value on **shape** anomalies: changes in rhythm, coupling or connectivity | E5 (primary), E4 |
| P2 | Topology adds **no** value on **volume** or **identity** anomalies: changes in rate or magnitude, or a specific host or port | E5 (primary), E4 |
| P3 | Published TDA results shrink under fair evaluation | E1 + E2, plus the literature audit |

E2 and E3 on real SMD anomalies are reported as **overall** results only. SMD does not label anomaly types, so those results do not enter P1 or P2.

## 2. Datasets and windows

**SMD (Server Machine Dataset; OmniAnomaly repository)**
- 28 machines, 38 metrics each, with the official train and test files.
- Training data is treated as normal; its labels are taken as zero.
- Validation is the **last 20% of each machine's training data**.
- **Windows:** 60 samples, step 10.
- **Window labels:**
  - **Positive:** at least one labelled point among the window's final 10 samples.
  - **Normal:** no labelled point anywhere in the window.
  - Otherwise the window is **excluded**.
- **Splits:** each machine is evaluated on its own test set. Error bars come from a bootstrap over the 28 machines.

**CTU-13 (labelled bidirectional NetFlow)**
- **Windows are per host**, built from the flows that host sends:
  - 10-second bins;
  - 60 bins per window (10 minutes);
  - a step of 30 bins.
- **Positive window:** it belongs to an infected host and contains at least one From-Botnet flow.
- **Normal window:** it belongs to a labelled normal host, contains at least one From-Normal flow, and contains no botnet flow.
- Hosts with only background traffic are excluded.
- **Splits:** leave-one-scenario-out. The validation set is one other training scenario, drawn by seed.
- **Behaviour types** are taken from the flow-label strings if those strings name behaviours (C&C, spam and so on). Otherwise each scenario gets the type of its listed behaviours, and **Mixed** if they disagree.

**Not used:**
- Exathlon: too heavy for the time available.
- Corrected CIC-IDS2017: no direct download.
- DARPA OpTC: about 1 TB, and the source papers released no code.

## 3. Anomaly types

**CTU-13 behaviours**

| Behaviour | Type |
|---|---|
| C&C | Shape |
| Port scan | Shape |
| P2P | Shape |
| Spam | Identity |
| Click fraud | Volume |
| DDoS | Volume |

**E5 injected anomalies (into SMD)**
- Injections go into **normal test segments only**. Training and validation data are never touched.
- **Eligible channels** are chosen from training data:
  - *periodic:* the autocorrelation at the dominant lag is above 0.5;
  - *coupled pairs:* the correlation |r| is above 0.7.
- A machine with no eligible channel is skipped for that type, and this is reported.

| Code | Type | Injection | Magnitudes swept (decision level in **bold**) |
|---|---|---|---|
| S1 | Shape | Change the period of a periodic channel by resampling the segment | factor 1.5, **2**, 3 |
| S2 | Shape | Phase-shift one channel of a coupled pair | 1/8, **1/4**, 1/2 period |
| S3 | Shape | Correlation break: replace a fraction of one coupled channel's segment with a random segment of the same channel. Its own marginal distribution is preserved | 0.5, 0.75, **1.0** |
| V1 | Volume | Level shift | +1σ, **+2σ**, +3σ |
| V2 | Volume | Spikes (3 points) | 3σ, **5σ**, 8σ |
| V3 | Volume | Variance scaling | ×1.5, **×2**, ×3 |

- **Duration:** 60 samples per injection.
- **Count:** 10 injections per type, per magnitude, per machine.
- **Spacing:** injections are at least 120 samples apart and never touch labelled anomalies.
- **Design note, stated openly:** shape injections keep each channel's mean and variance, and volume injections keep the coupling between channels. E5 is a mechanism test by design. The real-data tests are E2, E3 and E4.

## 4. Feature sets

All feature sets use the same windows. Scalers are fitted on training data only.

| Set | Contents |
|---|---|
| **F1 Simple** | Per channel: sum, mean, variance, unique count. CTU-13 adds: unique destination IPs, destination ports, flow count, bytes, packets |
| **F2 Simple-structure** | Correlation summary (mean, min, max, std of off-diagonal values; top-eigenvalue share). Periodicity (FFT peak share; maximum autocorrelation at lags 2–30). PCA reconstruction error, from PCA fitted on training data. CTU-13: destination graph components, cycle rank and degree statistics |
| **F3 Topology** | Delay embedding (m = 3, τ = 1). Vietoris–Rips persistence via `ripser`, H0 and H1. Summarised by: number of bars; total, max, mean and std of lifetimes; persistent entropy; Betti curve at 10 fixed multiples of the median pairwise distance. 32 features, no tuning |
| **F5 Fluent Bit-style** | Replicates the `tda` processor: log1p transform, embedding 3 × 1, a quantile radius scan that keeps the scale with maximum β1, then outputs β0, β1, β2 (3 features). **Exploratory** |
| **Z Zigzag (CTU-13)** | A per-host hypergraph per window, built from destination IP/port groups. Zigzag persistence across consecutive windows, summarised by bar statistics. **Z-control** = I03's control: edges, vertices, components, diameter |

## 5. Methods and detector

- **Our pipeline**
  - The detector is Isolation Forest with **fixed defaults** (300 trees, max_samples = auto, seed-varied), the same for every feature set.
  - There is **no tuning**, because SMD validation data contains no anomalies.
  - Training uses normal windows only.
- **Re-tested methods**
  - **TADA** (`tdaad`, default parameters).
  - **PHANTOM** (PNNL, default parameters, run on the same windows).
  - **TAME** (released code). Two changes only:
    - its test-prevalence threshold is replaced by our threshold rule;
    - its detector is fixed **in advance** to its Isolation Forest variant, instead of the best detector chosen on test data. All its detectors are still reported, separately.
- **Threshold:** set at a **1% false-alarm rate on validation windows**. It never uses test data.
- **Repeats:** 5 seeds.
- **Pinned versions:** library and repository commits are recorded with each run.

## 6. Metrics

- **Primary:** PR-AUC (average precision).
- **Also reported:**
  - recall at the 1% false-alarm threshold;
  - false alarms per day (consecutive alarms count as one);
  - ROC-AUC, to compare with TAME's published numbers;
  - runtime per window (single CPU thread, hardware recorded).
- **Point adjustment:** none.

## 7. Experiments

| ID | Description |
|---|---|
| **E1** | TAME as released, unchanged, compared with its reported SMD ROC-AUC (0.808). **Timebox: 2 days.** If it cannot be run, we report the failure with logs |
| **E2** | Fair re-test on SMD of TADA, PHANTOM and TAME (fixed as in §5), alongside our F1+F2 and F1+F2+F3 pipelines |
| **E3** | Swap test on SMD with the same detector. Feature sets: F1, F1+F2, F1+F3, F1+F2+F3, F5 |
| **E4** | CTU-13: F1+F2 vs F1+F2+F3, and Z vs Z-control. **Zigzag timebox: 3 days.** The fallback is a sequence of per-window diagrams with Wasserstein distances between consecutive windows. This re-implements I03's idea; it is not an exact replication |
| **E5** | SMD injections (§3). For each type: F1+F2 vs F1+F2+F3, plus detection-vs-magnitude curves |

## 8. Decision rules

**Comparisons**
- **Main gain:** Δ = PR-AUC(F1+F2+F3) − PR-AUC(F1+F2).
- **Swap gain:** Δs = PR-AUC(F1+F3) − PR-AUC(F1+F2).
- **Confidence interval:** 95% cluster bootstrap, 2,000 resamples, over machines (SMD) or scenarios (CTU-13), after averaging across seeds.

**Verdicts**

| Verdict | Condition |
|---|---|
| **Helps** | Δ > 0 and the CI excludes 0 |
| **No added value** | The CI lies inside ±0.02 |
| **Inconclusive** | Anything else, including cells with fewer than 20 positive or fewer than 20 normal test windows |

**Per prediction**
- **P1**
  - **Supported:** Δ (main) "helps" for at least 2 of the 3 shape types (S1–S3) at the decision magnitude.
  - **Refuted:** "no added value" for at least 2 of the 3.
  - CTU-13 shape cells are reported alongside as supporting evidence.
- **P2**
  - **Supported:** all of V1–V3 are "no added value" at the decision magnitude.
  - **Refuted:** any volume type "helps" with Δ ≥ 0.02 after Bonferroni correction (α = 0.05/3).
- **P3**
  - **Supported:** TAME's ROC-AUC or PR-AUC under E2 is lower than its E1 value, with the bootstrap CI of the difference excluding 0.
  - **Refuted:** the difference lies within ±0.02.
  - **No verdict:** if E1 fails. P3 then rests on the literature audit alone.
- **Not used for any verdict:** F5, the TADA results and the PHANTOM results. They are reported descriptively.

## 9. Deviations and timeboxes

- Every change made after data access is logged with its date, its reason and the results it affects. Original and amended results are both reported.
- Any analysis added later is labelled exploratory.
- **Timeboxes:**
  - E1: 2 days.
  - Zigzag: 3 days.
  - Any re-tested repository that fails to run: 2 days. After that, the failure is reported with logs.

## 10. To verify at data access (logged, does not change the rules)

1. CTU-13: the label strings, behaviour names, infected and normal host IPs, and the number of normal hosts in each scenario.
2. SMD sampling interval, and the train/test lengths for each machine.
3. Commits to pin for `tdaad`, TAME, PHANTOM and Fluent Bit (`tda.c`), plus the default parameters of each.
4. Whether the zigzag library is available.
5. Hardware and library versions.

## Change log (v2 → v3)

| # | Change | Reason |
|---|---|---|
| 1 | Exathlon replaced by **SMD** as the systems dataset | Time; SMD is used by most TDA papers, so results compare directly |
| 2 | CTU-13 windows changed to **per host** | Network-level windows would make almost every window positive |
| 3 | **Swap test** (F1+F3 vs F1+F2) added | Matches our own R2 criterion |
| 4 | **E5 injections** made the primary test of P1 and P2 | SMD has no anomaly types, and the real cells are underpowered |
| 5 | TADA rerun on Exathlon dropped; P3 now uses TAME (E1 vs E2) | The original TADA protocol cannot be reconstructed |
| 6 | PHANTOM added as a re-tested method; zigzag test (Z) added | Covers the PNNL topology-over-time lineage |
| 7 | Isolation Forest uses fixed defaults; tuning grid removed | SMD validation has no anomalies, so selection by PR-AUC is impossible |
| 8 | Exathlon type table (T1–T6) removed | Dataset not used |
