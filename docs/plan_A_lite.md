# Plan A-lite: exploratory follow-ups (frozen 2026-11-21, before any run)

All of it is **exploratory and outside the prereg**. No prediction (P1–P3) is decided from it. E1–E5 stay as reported.
No design change after results. Code: `src/alite.py`, `src/alite_l1.py`, `src/knn_detector.py`, `scripts/*`. Configs: `configs/l*.yaml`.
Words "Helps / No added value / Inconclusive" are the §8 rules applied for orientation only.

## L1 Metric table (no new fitting choices)
Per-window scores are **not stored** in E3/E2, so each detector is **re-scored with the committed code, same seeds, same config**
(hooks `on_scores` in `src/runner.py` and `src/e2_methods.py` change no result). `scripts/summarize_l1.py` compares PR-AUC row by
row with `results/E3`, `results/E2m`, `results/E2m_tame`; it reports the largest difference and the number of rows that differ by more
than 1e-9. TAME-IF runs in `.venv-tame`; a BLAS-thread difference of about 0.001 was seen for TAME before, so any TAME
mismatch is reported as it is, not hidden.
- **Rows:** F1, F1+F2, F1+F3, F1+F2+F3, F5 (E3 config); TADA, PHANTOM, TAME-IF (E2 configs); **RANDOM** (uniform scores, seed 0, same
  1%-validation threshold rule, one seed).
- **Columns:** PR-AUC, ROC-AUC (ours, unchanged); **F1** and **accuracy** at our validation threshold; **best-F1** over **400
  thresholds chosen on test labels** (evenly spaced from the lowest to the highest test score; paper style, decides nothing);
  **point-adjusted best-F1** at window level (same 400 thresholds).
- **Point adjustment:** a run of consecutive windows labelled positive counts as fully alarmed if any window in it alarms. A normal or
  excluded window ends a run. Excluded windows (label −1) are dropped from every count.
- **Mean over machines** with at least one positive window; per machine, mean over the 5 seeds; 95% cluster bootstrap CI (2,000
  resamples). Deterministic methods (PHANTOM, OCSVM, kNN) repeat the same numbers.

## L2 H1 fairness check
**(a) Synthetic.** 10 channels, even channels period 24, odd channels period 37, amplitude 1, random phase, noise sd 0.1. Train 2,400
samples (80% fit, 20% validation), test 2,400 with one anomaly: channel 0 runs at period 48 (×2) for 60 samples from t = 1,200.
Same windows and label rule as SMD (60 / 10 / tail 10). Features: the 16 H1 summaries of F3 on the whole z-scored window (fitted on
training windows), m = 3, **τ = 1 and τ = 6**. Reported: mean maximum H1 persistence on positive and on normal test windows, and
H1-only Isolation Forest (300 trees) ROC-AUC and PR-AUC. **5 synthetic datasets** (seeds 0–4, the same seed for the forest); mean and sd.

**(b) SMD, "H1-friendly".** Same windows, labels, split, threshold, 5 seeds, Isolation Forest defaults as E3. Feature sets: `F1+F2`,
`F1+F2+PC-H0`, `F1+F2+PC-H1`.
- **τ per machine** (training windows only): non-overlapping training windows (every 6th) are joined per channel; centre each
  channel; autocorrelation at lags 0..16; skip constant channels; average over channels; **first local minimum in [2, 15]**
  (`acf[k] < acf[k−1]` and `acf[k] ≤ acf[k+1]`). If there is none: the lag of the lowest value in [2, 15]. If every channel is
  constant: 1.
- **Per-channel clouds:** each channel is z-scored with the training mean and std (std floor 0.01, as F3) and delay-embedded alone
  (m = 3, τ = τ_ac) into a 3-D cloud of 60 − 2τ points; ripser H0/H1; the same 16 summaries per dimension as F3; Betti scales =
  0.2 … 2.0 × **that channel's** training median pairwise distance (1 if it is 0). The 38 per-channel vectors are **averaged**.
- **Comparisons (PR-AUC, paired, CI over machines):** `F1+F2+PC-H1` vs `F1+F2`, `F1+F2+PC-H0` vs `F1+F2`, `H1` vs `H0`.
- **Rule fixed now:** if `F1+F2+PC-H1` vs `F1+F2` is **"Helps"**, the report says **"loops help with a proper embedding"**; otherwise
  **"H1 adds nothing even with an H1-friendly embedding"**. (Overlap case "CI above 0 and inside ±0.02": reported as in the V2 gap,
  and the sentence follows the word "Helps" only if the table-order rule gives "Helps".)
- `F1+F2` is the same as E3's, so it must reproduce E3's PR-AUC exactly; the summary prints the largest difference.

## L3 kNN detector
k = 5, mean Euclidean distance to the 5 nearest **training** windows, `StandardScaler` fitted on training features. Deterministic: one
run (seed 0). Feature sets: F1, F1+F2, F1+F3, F1+F2+F3, F1+F2+F3-H0, F1+F2+F3-H1 (the halves of E3's F3, as E3x). Reported next to the
Isolation Forest values of `results/E3` and `results/E3x`: **Δ main** (F1+F2+F3 − F1+F2), **Δs** (F1+F3 − F1+F2), **ΔH0**, **ΔH1**
(each of the halves added to F1+F2, minus F1+F2), PR-AUC, 95% CI over machines.

## L4 MST
No new experiment. `scripts/l4_mst.py` applies the existing MST-versus-H0 comparison to **every test window of machine-1-1** (F3
cloud, m = 3, τ = 1): the largest absolute difference between sorted MST edge lengths and sorted finite H0 deaths (zero-length
edges ignored on both sides), and the time per window of the MST path and the ripser (maxdim 0) path, both including the distance
matrix, single thread, best of 3 passes.
Found while writing it: `src/live.py` (the Streamlit Live page) passed the distance matrix to scipy's MST as a **dense** array, and scipy
drops values below about 1e-8 there, so a window with two identical points could give a wrong tree. Now passed as a sparse matrix
(`src/live.py`, `src/alite.py`). Display only; no result file used it.

## L5 Self re-check (step 4)
`literature/recheck/blank_sheet.csv`: C01, C02, H04, I03, D04, N10, B03 with title, authors, year, link and empty R1–R13, OUT, ISO.
It carries **no score, group, outcome or note**. `literature/recheck/definitions.md` holds the rubric wording (R1–R13, OUT, ISO) as supplied by Anirudh. `scripts/agreement.py <filled sheet>`: % agreement and Cohen's kappa against `tda_records_master.csv`, per
column, per paper, and pooled (R1–R13, OUT, ISO, all). Free text in OUT/ISO is cut to its first word on both sides.

## Step 2 (after the commit)
`python scripts/run_l1.py --group ours --jobs 4 --resume`, `--group methods --jobs 4 --resume`, `.venv-tame/bin/python
scripts/run_l1.py --group tame --jobs 2 --resume`, `python scripts/summarize_l1.py`;
`python scripts/run_l2a.py`; `python scripts/run_experiment.py configs/l2b_h1_friendly.yaml --jobs 4 --resume` then
`scripts/summarize_l2b.py`; `python scripts/run_experiment.py configs/l3_knn.yaml --jobs 4 --resume` then `scripts/summarize_l3.py`;
`python scripts/l4_mst.py`.
