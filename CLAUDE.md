# Project brief for Claude Code: read this first

**Project:** *Does Topology Help? An Audit and Fair Re-evaluation of Topological Data Analysis (TDA) Anomaly Detection for Systems and Security Telemetry*

**Owner:** Anirudh (solo, semester project). Final evaluation is in about 3–4 weeks from 2026-11-19 and consists of a report, a presentation and, ideally, a working demo. A paper may follow later.

**Working style:** Anirudh must be able to explain every step in the viva.
- Explain what you are doing and why, briefly.
- Prefer simple, readable code over clever code.
- Keep token use low: no long recaps.

## 1. What the project is

- **The claim being tested:** published papers say TDA features (persistent homology, Mapper, zigzag) improve anomaly detection on security and systems telemetry.
- **What our audit found:**
  - Of the 32 core papers (D03 excluded), only 4 compare topology fairly against cheap features. The results: 2 lost, 1 tied, 1 weak win.
  - Most "wins" either only remove topology or compare against other people's systems.
  - Benchmarks are often flawed (KDD99, CIC-IDS2017, SMD point-adjust).
  - Three released repositories use test information when choosing thresholds.
  - No paper tests against an adversary.
  - Systems telemetry is barely covered.
  - Fluent Bit ships a `tda` processor with no evaluation at all.
- **Our job now:** re-test the claims fairly, using public code and data. **All rules are frozen in `preregistration/preregistration_v3.md`. Follow it exactly.**

## 2. Hard rules

1. **Do not open or download any data** until `preregistration_v3.md` is committed **alone** and tagged `prereg-v3`.
2. **No test information anywhere.**
   - Scalers, PCA and every other fitted transform are fitted on training data only.
   - Thresholds are set at 1% false alarms on **validation** windows.
   - No point adjustment.
   - No model or detector selection on test data.
3. **Any change after data access** goes in `deviations_log.md` with the date, reason and affected results. Both the original and the changed result are kept.
4. **Timeboxes:**
   - TAME reproduction (E1): 2 days.
   - Zigzag library or features: 3 days, then use the fallback in prereg §7.
   - Any repository that won't run: 2 days, then report the failure with logs.
5. **Every result file records** the git commit, the seed and the library versions.
6. **Never invent numbers.** If something fails, report that it failed.
7. **Claude never runs `git add`, `git commit`, `git tag` or `git push`.** At the end of each task, Claude gives Anirudh the exact commands to run: which files to `git add` and a short commit message. Never stage `data/`, `external/` or `.venv/`.

## 3. Plan and status

| Cluster | Step | Status |
|---|---|---|
| A | 1. Prereg v3 written | ✅ |
| A | 2. Commit prereg alone, tag `prereg-v3`, push | ⬜ (Anirudh) |
| B | 3. Exclude D03, fix the Fig 1 footnote, regenerate figures | ✅ (`literature/`) |
| B | 4. Self re-check: re-score 7 papers blind, report % agreement | ⬜ (Anirudh) |
| B | 5. Decision log | ✅ draft in `docs/decision_log.md`; Anirudh fills in the dates |
| C | 6. Data: SMD (OmniAnomaly GitHub repo, `ServerMachineDataset`) and CTU-13 (`.binetflow` from the Stratosphere archive, on the laptop). Record checksums in `data/README.md` | ⬜ |
| C | 7. Environment: Python 3.10/3.11 venv; numpy, pandas, scikit-learn, ripser, persim, gudhi, matplotlib, streamlit; `tdaad` (github.com/IRT-SystemX/tdaad); clone TAME (github.com/SPIRE-Lab-2025/TAME) and PHANTOM (github.com/pnnl/PHANTOM) into `external/`; zigzag via `dionysus`, with the fallback if needed. Pin versions | ⬜ |
| C | 8. Shared pipeline in `src/`: loaders, windowing, splits, threshold, metrics, bootstrap CI, one config per experiment in `configs/`. Unit-test it on toy data | ⬜ |
| D | 9. F1 simple and F2 simple-structure features (incl. PCA error) | ⬜ |
| D | 10. F3 topology: delay embedding 3×1, ripser H0/H1, 32 summary features | ⬜ |
| D | 11. F5 Fluent Bit-style β0/β1/β2 (logic from `plugins/processor_tda/tda.c`; pin the commit) | ⬜ |
| D | 12. Zigzag and Z-control features for CTU-13 | ⬜ |
| E | 13. **E2** fair re-test on SMD: TADA, PHANTOM, TAME (fixed) | ⬜ |
| E | 14. **E3** swap test on SMD | ⬜ |
| E | 15. **E5** shape vs volume injections | ⬜ |
| E | 16. E1 TAME as released vs reported SMD ROC-AUC 0.808 | ⬜ |
| E | 17. E4 CTU-13 zigzag re-test, per-host windows | ⬜ |
| F | 18. Streamlit demo `app.py` reading precomputed `results/` | ⬜ |
| G | 19–21. Report, slides, AI-use note | ⬜ |

**Order:** A → B and C → D → E (E2, E3, E5 first) → F → G.

**Minimum if time is short:** A, B, E2, E3, E5, the report and the slides.

## 4. Key facts already verified (do not redo)

- **TAME** @ `e77c0d9`:
  - When validation has no anomalies, the threshold falls back to the test anomaly rate: `utils/experiment_core.py` L105–112 and L216.
  - "TAME (best)" is the maximum over detectors chosen on test AUROC: `consolidate.py` L66–71.
  - On SMD, training labels are all zero (`utils/data_loaders.py`).
- **TopoGDN** @ `2b8df3b`:
  - The default `-report best` searches thresholds using test labels: `main.py` L349; `evaluate.py` L129–143. This behaviour is inherited from GDN.
  - Not re-tested: the build is broken and it needs a GPU.
- **MSF-IDS** @ `bb8ccf5`: assigns random timestamps on the NetFlow datasets (`Data/datasets_NF.py` L57–71), and those timestamps are then used as the PH filtration.
- **PHANTOM** @ `8caa25a`:
  - BSD licence.
  - Ripser H0 only.
  - Scores with the Wasserstein distance (`main.py` calls `compare(..., 2)`) between the baseline diagram and the baseline plus the new window, with a rolling baseline update. (`AnomalyScore.compare` defaults to bottleneck, but `main.py` never uses the default.)
- **Fluent Bit `tda` processor:**
  - Metrics only.
  - Transform: sign·log1p.
  - Delay embedding with `embed_dim=3`, `embed_delay=1`.
  - Rips with a quantile radius scan that keeps the scale with maximum β1.
  - Outputs β0/β1/β2 gauges.
  - Defaults: `window_size` 60, `min_points` 10, `threshold` 0.
  - No published evaluation. PR fluent/fluent-bit#11250.
- **SMD:** 28 machines, 38 metrics, official train/test/labels.
- **CTU-13:**
  - 13 scenarios, labels in `.binetflow`: Background, From-Botnet, C&C, From-Normal.
  - The Stratosphere site is reachable from the laptop, not from cloud sessions.

## 5. Repo layout

```
CLAUDE.md                 this brief
preregistration/          frozen plan (do not edit after tag; use deviations_log.md)
deviations_log.md         changes after data access
docs/                     decision_log.md, code_verification.md, paper_dossier.pdf
literature/               tda_records_master.csv (95 records), make_figs.py, figures/
src/                      pipeline code
configs/                  one YAML per experiment
results/                  outputs (CSV/Parquet + run metadata)
external/                 cloned third-party repos (gitignored)
data/                     datasets (gitignored) + README with sources/checksums
app.py                    Streamlit demo (later)
```

## 6. Report structure (the final deliverable)

1. Problem and question.
2. Background: what TDA can and cannot capture.
3. Method: search, screening, rubric, self re-check, preregistration, fair rules.
4. Audit results: Figures 1–2 and the dossier summary.
5. Code-audit findings.
6. Experiments E1–E5.
7. Why: shape vs volume.
8. Limitations and future work: attacker testing, OpTC replication, analyst studies.
9. Appendix: decision log and the full paper table.
