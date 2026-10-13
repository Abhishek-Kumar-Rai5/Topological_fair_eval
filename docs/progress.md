# Progress

Updated at the end of every task. Latest entry first.


## 2026-11-20 (late): E4 typing by flow-label strings, E4x and E3x built (not run)

**Done**
- **Primary E4 typing = per positive window from the flow-label strings** (prereg §2 first rule): `CC<n>`, `P2P` = Shape, `SPAM` = Identity, several = Mixed, none = Unnamed (note 61, note 67). **Sensitivity typing** = README then Table 2 per scenario (IRC = C&C, HTTP not counted): Shape S3, S6, S8, S12; Mixed S1, S2, S4, S5, S7, S9, S11, S13; no Volume-only (data limitation). S7 stays Mixed (README wins; conflict in note 61). The strings cannot name port scan, DDoS or click fraud, so neither typing can test P2 and neither decides a prediction.
- Exploratory **E4x** (CTU-13) and **E3x** (SMD): F3 split into H0-only and H1-only (note 68, logged in `deviations_log.md`). `src/features_topo.py`, `src/feature_sets.py` (groups `F3-H0`, `F3-H1`), `src/e4.py`, `configs/e3x_f3_split.yaml`, `configs/e4_f3_split.yaml`, `scripts/summarize_e3x.py`, `scripts/summarize_e4x.py`. 138 tests pass. Nothing run yet.

**Next** (after your commit): E4 (`scripts/run_e4.py configs/e4_ctu13.yaml --jobs 4 --resume`, `scripts/summarize_e4.py results/E4`), then E4x (`--stage 2 ... --out results/E4x`, `scripts/summarize_e4x.py`), then E3x (`scripts/run_experiment.py configs/e3x_f3_split.yaml --jobs 4 --resume`, `scripts/summarize_e3x.py results/E3x`).

**Open decisions (Anirudh)**
- None new. Still open: TBD dates in `docs/decision_log.md`; `prereg-v3` tag on the remote; Step 4 self re-check.

## 2026-11-20: E4 on CTU-13 built (worktree `dev`; nothing run beyond one dry run)

**Done**
- Open details recorded **before** coding: notes 49-66 in `docs/implementation_notes.md`.
- `src/ctu13.py` (flows, host windows, labels, graphs), `src/features_ctu13.py` (graph features, Z-control, Z), `src/e4.py` (stage 1 table per scenario, stage 2 leave-one-scenario-out folds), `scripts/run_e4.py`, `scripts/summarize_e4.py`, `configs/e4_behaviours.yaml`. 16 new tests, all pass.
- **Zigzag works** with `dionysus` 2.2.3 (already installed, no system package). The 3-day timebox started 2026-11-20; the §7 fallback is not needed. Z at window t uses only windows <= t (tested against a truncated recomputation).
- Dry run: scenario 7 stage 1 only (114k flows, 0.4 s): 16 windows (2 positive, 14 normal), no NaN. Stage 2 not run (needs all 12 scenarios).

**Next**
1. Merge master into dev if not yet done (`summarize_smd.py`), commit, then run E4 in the main folder.
2. After the run: check `n_windows_edge_capped` in the `by_scenario/*.meta.json` files (graph cap, note 56).

**Open decisions (Anirudh)**
- Note 61: scenario behaviour types are from README text; six scenarios are Unknown. Shape and Volume cells have only 2 scenarios each (verdict will probably be Inconclusive).
- Notes 53, 56, 59: raw-count channels, 2,000-edge cap, trailing 5 windows for Z.
- Scenario 7 has only 2 positive windows, and several scenarios are short: per-fold PR-AUC will be noisy.




## 2026-11-20 (night): E5b run (exploratory)

**Done**
- E5b run at commit `c6b2915` (clean tree, `git_dirty` false), 28 machines × 5 seeds, 191 s with `--jobs 4`. `results/E5b/` (`E5b_summary/verdicts.csv`, `E5b_curves.png`, `E5b_placement.csv`). Summary: `scripts/summarize_e5b.py`.
- **3 of 18 cells count** (baseline ROC-AUC at least 0.7): V1 at 2 and 3 sigma (No added value) and V2 at 8 sigma (Helps, Δ = +0.021 [0.014, 0.029]). The other 15 cells are marked "Baseline fails". All shape cells (S1, S2, S3) fail, so **E5b cannot test P1**. As agreed before the run, nothing was changed afterwards, and P1/P2 are not decided from E5b.
- Skipped (fewer than 5 usable channels): S1 on 6 machines, S2 on 8 machines (placement file lists them).

**Next**
- E4 CTU-13 + zigzag; dashboard; report.

**Open decisions (Anirudh)**
- Still open: TBD dates in `docs/decision_log.md`; `prereg-v3` tag on the remote; Step 4 self re-check.

## 2026-11-20 (later): P3 CI, V2 gap, E5b built (not run)

**Done**
- P3 with CIs over 28 machines (`scripts/p3_e1_vs_e2.py`, `results/P3/`): all four comparisons Refuted under §8 (see below in the E1 entry).
- V2 precedence gap logged in `deviations_log.md`; E5 P1/P2 reported as uninformative (baseline at chance); mechanical verdicts only as a footnote.
- **E5b (exploratory, k = 5, approved)**: `src/injections_k.py`, `src/e5b.py`, `scripts/summarize_e5b.py`, `configs/e5b_injections.yaml`, dispatch in `scripts/run_experiment.py`, `tests/test_e5b.py` (7 tests). **Not run yet**: the code is to be committed first so the run is on a frozen, clean tree. Rule: a cell counts only if the F1+F2 ROC-AUC is at least 0.7; P1/P2 are not decided from it; if the baseline still fails, report and stop.

**Next**
1. After your commit: `python scripts/run_experiment.py configs/e5b_injections.yaml --jobs 4 --resume`, then `python scripts/summarize_e5b.py results/E5b`.
2. E4 CTU-13 + zigzag; dashboard; report.

**Open decisions (Anirudh)**
- Still open: TBD dates in `docs/decision_log.md`; `prereg-v3` tag on the remote; Step 4 self re-check.

## 2026-11-20: E5, E2 methods, TAME (own windows), E1

**Done** (all runs from commit `ac1524e`, tracked files clean, so `git_dirty` is false in every result sidecar; four laptop shutdowns were handled with `--resume` / cached chunks)
- **E5** (28 machines × 6 types × 3 magnitudes × 5 seeds): `results/E5/` with `E5_summary/verdicts/predictions.csv`. **The F1+F2 baseline is near chance in every cell (ROC-AUC 0.50–0.60, PR-AUC 0.03–0.05, base rate 0.03), so the preregistered E5 failed to detect the injections and its P1/P2 verdicts are uninformative.** No bug found: the injections change the windows and detection rises with magnitude.
- **E2:** TADA, PHANTOM (`results/E2m/`), TAME on our windows (`results/E2m_tame/`, `.venv-tame`), fixed TAME on its own windows (`results/E2_tame_own/`). Table: `results/E2m/E2_table.csv` (`scripts/summarize_e2.py`). SHA-256 of `run_tame_own_windows.py` and `summarize_e2.py` are in their `.meta.json` files.
- **P3 (2026-11-20):** `scripts/p3_e1_vs_e2.py`, `results/P3/`. Fixed TAME-IF on its own windows minus E1, bootstrap over 28 machines: vs E1 best (GMM) ROC-AUC −0.0068 [−0.0180, 0.0033], PR-AUC −0.0029 [−0.0199, 0.0152]; vs E1 Isolation Forest ROC-AUC −0.0011 [−0.0029, 0.0007], PR-AUC −0.0016 [−0.0058, 0.0029]. All four inside ±0.02, so §8 gives **Refuted** for each. BLAS thread count changes TAME's IF AUROC by about 0.001 (1 vs 4 threads).
- **E5:** P1 and P2 are reported as **uninformative** because the baseline is at chance. Mechanical verdicts (footnote only): P1 Refuted, P2 Inconclusive. V2 gap in §8 logged in `deviations_log.md`.
- **E1:** TAME's released SMD notebook, unchanged, in `external/TAME_e1`: best detector GMM, ROC-AUC 0.808 (reported 0.808). Isolation Forest 0.801. `results/E1/`. The notebook's plotting cell failed on a newer matplotlib after the results were written (details in implementation notes).
- `CLAUDE.md` §4: PHANTOM now says Wasserstein. `.gitignore`: `.venv-tame/`. Choices 39, 40, 43 approved and logged (39 in `deviations_log.md`).

**Next**
1. E5b (exploratory) only after your OK; proposal is in `deviations_log.md`.
2. E4 CTU-13 + zigzag; dashboard; report.

**Open decisions (Anirudh)**
- E5b: run or not.
- P3: the E1 vs E2 ROC-AUC difference is −0.008 (E1 best-of-detectors 0.808, fixed TAME-IF on its own windows 0.800), inside ±0.02. A formal CI of that difference was not computed.
- Still open: TBD dates in `docs/decision_log.md`; `prereg-v3` tag on the remote; Step 4 self re-check.

## 2026-11-19 (night): E5 injections, atomic writes, E2 adapters (worktree `dev`; nothing run on real data)

**Done**
- **Atomic writes + validated `--resume`** (`src/metadata.py`, `scripts/run_experiment.py`, `tests/test_atomic.py`). Done before any real run, so E5 and the E2 adapters build on it.
- **E5 injection module** (`src/injections.py`) and **runner** (`src/e5.py`, `scripts/run_experiment.py configs/e5_injections.yaml`), **summary** (`scripts/summarize_e5.py`, reuses `summarize_smd.py` from master). Choices 22–38 in `docs/implementation_notes.md`.
- Removed the older, duplicate `src/inject.py` / `tests/test_inject.py`; its only extra rule (training std ≥ 0.01 for V1–V3) was ported.
- **E2 adapters** (`src/e2_methods.py`; `run_experiment.py --methods`): TADA, PHANTOM, TAME (IF fixed; GMM and OCSVM reported separately). Choices 39–48 in `docs/implementation_notes.md`. 108 tests pass (1 skipped: needs giotto-tda).
- Toy-data tests only.

**Next**
1. Merge `master` into `dev` (`summarize_smd.py` is needed by `summarize_e5.py` and its test).
2. Commit (3 batches, commands in chat), then run E5 and E2 methods in the main folder.
3. TAME: giotto-tda is not installed; installing it in the main venv would downgrade numpy and scikit-learn. Use a separate venv (note 48).

**Open decisions (Anirudh)**
- E5: nothing new. Choices 22–25 confirmed in chat today.
- **E2 choices 39–48 need your OK**, mainly: 39 (TAME on our windows, a third change), 40 (TADA window mapping), 43 (PHANTOM vector), 48 (TAME venv).
- **CLAUDE.md §4 says PHANTOM uses the bottleneck distance; `main.py` uses Wasserstein** (note 42). Please correct the line.
- Still open: TBD dates in `docs/decision_log.md`; check `prereg-v3` is on the remote; Step 4 self re-check.





## 2026-11-19 (night, later): ROC-AUC added, exploratory Δ against F1

**Done** (from existing E3 results; nothing rerun)
- `scripts/summarize_smd.py` now also reports ROC-AUC (mean [95% CI]); E3 re-summarised. The PR-AUC numbers and §8 verdicts are unchanged.
- Checked: false alarms per day merges consecutive false-alarm windows into one event (`src/metrics.py` `count_alarm_events`); the sampling interval is `minutes_per_point: 1.0` in `configs/e2_fair_retest.yaml` and `configs/e3_swap.yaml`, passed by `src/runner.py` to `evaluate`. F3 embeds the whole 38-channel window jointly (one 114-dim point cloud per window). F5 outputs 3 features.
- **EXPLORATORY, not in the prereg** (`results/E3/E3_exploratory.csv`): paired cluster-bootstrap Δ PR-AUC against F1: F1+F2 −0.021 [−0.035, −0.006]; F1+F3 +0.047 [0.025, 0.072]; F1+F2+F3 +0.034 [0.012, 0.056]. Main Δ (F1+F2+F3 − F1+F2) is above 0 on 23 of 28 machines (4 below, 1 equal). No verdicts are attached to these.

**Open decisions (Anirudh)**
- Earlier items still open: fill TBD dates in `docs/decision_log.md`; check the `prereg-v3` tag is on the remote; Step 4 self re-check; harden `--resume` and file writes.

## 2026-11-19 (night): E3 and E2 (our pipelines) run on SMD

**Done**
- E3 (5 feature sets) and E2 our pipelines (F1+F2, F1+F2+F3): 28 machines × 5 seeds, all at commit `9c28f8f`, `git_dirty` false. The laptop shut down twice mid-run; `--resume` continued from the saved machines (one half-written machine, 2-1, was deleted and rerun).
- `scripts/summarize_smd.py` applies §6/§8 and writes `results/<exp>/<exp>_summary.csv` and `_verdicts.csv`.
- Verdicts (prereg §8, PR-AUC): E3 Δ = +0.054 [0.035, 0.076] Helps; Δs = +0.067 [0.041, 0.097] Helps; E2 Δ identical to E3. Details in the summary files. Not interpreted yet.

**Next**
1. Adapters for TADA, PHANTOM, TAME (rest of E2), E1 (TAME as released, 2-day timebox), E5 injections, E4 CTU-13 + zigzag.
2. Harden `--resume` and file writes against crashes (write to a temp file then rename; skip only valid files).

**Open decisions (Anirudh)**
- Earlier items still open: fill TBD dates in `docs/decision_log.md`; check the `prereg-v3` tag is on the remote; Step 4 self re-check.

## 2026-11-19 (evening): Step 11 F5 and the runner

**Done**
- Decisions applied: behaviour types for E4 (README fallback, not a deviation), infected hosts confirmed, std floor 0.01 accepted, E4 limitation (7 distinct normal hosts) logged as a data fact.
- **F5** (`src/features_fluentbit.py`, registered in `feature_sets.py`): copy of the `tda.c` logic at `25ea7e1`, literally: every channel first-differenced (dt = 1 sample), then signed log1p, one group per channel. A first "gauges, no differencing" version was withdrawn before any results. Unit-tested against a hand-worked example (unit square). 60 tests pass for the features.
- **Finding:** the Fluent Bit wrapper counts *finite* bars with persistence ≥ 1e-3, not infinite bars as its comment says. So loops still alive at the chosen scale are not counted. Written up in `docs/code_verification.md` §4. I fetched the 4 extra files it needs into `external/fluent-bit/` (hashes in `docs/external_versions.md`).
- **Runner** (`src/runner.py`, `scripts/run_experiment.py`): config → per-machine CSV + final CSV, each with a `.meta.json` (git commit, seeds, config, library versions). Features are computed once per machine, the forest is refit per seed. `--jobs`, `--resume`, `--machines` options. 66 tests pass. Smoke-tested on toy data only; **E2/E3 have not been run on SMD.**

**Next**
1. Your "go" to run E3 on SMD (all 5 feature sets) and the our-pipelines part of E2. F5 is the slow part (about 40 ms per window; about 1.5 h single-threaded, about 15 min with `--jobs 8`). **Commit first**, so `git_dirty` is false in the result metadata.
2. Adapters for TADA, PHANTOM, TAME (E2) and the injection code (E5); E1 (TAME as released, timebox 2 days).

**Open decisions (Anirudh)**
- (Resolved) F5 choices 16–21 confirmed.
- Run E3 and E2 (our pipelines) now? Which `--jobs` value fits your laptop?
- Still open: fill TBD dates in `docs/decision_log.md`; check the `prereg-v3` tag is on the remote; Step 4 self re-check.

## 2026-11-19 (later): F3 decisions applied, CTU-13 scenario 10 excluded, CTU-13 facts

**Done**
- F3 rewritten per your decisions: z-score with training mean/std (std floor 0.01), Betti scales 0.2–2.0 × the training median distance (once per machine), finite bars only. Tests updated: 50 pass. Real-data check (2 machines): all finite, about 1.2 ms per window.
- `docs/implementation_notes.md` F3 section updated (choices 12–15 now confirmed). `deviations_log.md` has 4 F3 clarifications and the scenario 10 exclusion.
- `scripts/ctu13_facts.py` writes `docs/ctu13_facts.md` (label strings, infected IPs, normal hosts per scenario).

**Findings (CTU-13)**
- 12 usable scenarios. 1 infected IP in most scenarios, 10 in scenario 9, 3 in scenarios 11 and 12.
- Only 5–7 normal hosts per scenario, and only 7 distinct normal IPs across all 12 scenarios (4 appear in every scenario). Normal-host windows are therefore not independent across scenarios, which limits E4.

**Next**
1. Step 11: F5 Fluent Bit-style features from `external/fluent-bit/tda.c`.
2. Runner script (config → results, with run metadata), then E2 and E3 on SMD.

**Open decisions (Anirudh)**
- Behaviour types for E4 (prereg §2): the botnet label strings name protocols or traffic types (DNS, SPAM, CC1, HTTP-Ad, Binary-Download), not single behaviours. Decide whether to use them as behaviours or to give each scenario a type from its README (Mixed if they disagree). Log it before E4 runs.
- Scenario 12 and 3 have From-Botnet flows sent by non-listed IPs; the plan treats only README-listed IPs as infected hosts. Confirm.
- Earlier items still open: fill TBD dates in `docs/decision_log.md`; check the `prereg-v3` tag is on the remote; Step 4 self re-check.

## 2026-11-19: status check after the laptop shutdown

**Done**
- Checked the repo state. `pytest -q`: 46 passed. `scripts/check_env.py`: ALL OK.
- Found the git repo damaged: the Step 10 commit `f458c4a` and 6 other loose objects are empty (0 bytes). The files on disk are intact. Last healthy commit: `0cf1360`.
- Plan status: Steps 1, 3, 6, 7, 8, 9 done. Step 10 code and tests done, but its commit was lost. Steps 11–21 not started. Steps 2 (push), 4 and 5 (dates) are partly or fully yours.

**Next**
1. Repair git: delete the empty objects, point `master` back at `0cf1360`, `git reset`, then recommit the 4 Step 10 files (commands given in chat).
2. Step 11: F5 Fluent Bit-style β0/β1/β2 features.
3. Runner script from config to results (with run metadata), then E2 and E3 on SMD.

**Open decisions (Anirudh)**
- Confirm or change F3 choices 12–15 in `docs/implementation_notes.md`. A change after results exist goes in `deviations_log.md`.
- CTU-13 scenario 10 (capture 51) has no `.binetflow`, so only 12 of 13 scenarios are usable. Decide how E4 handles this (needs a `deviations_log.md` entry if it differs from the prereg).
- Fill the TBD dates in `docs/decision_log.md`.
- Check that the tag `prereg-v3` is on the remote (`git ls-remote --tags origin`). `origin/master` is still at `773d00d`.
- Step 4: self re-check of 7 papers.
