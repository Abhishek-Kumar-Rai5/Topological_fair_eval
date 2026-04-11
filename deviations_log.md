# Deviations log

Changes made after data access (date | change | reason | affected results).

| Date | Change | Reason | Affected |
|---|---|---|---|
| 2026-11-19 | **Clarification (before any results)** of prereg §8 "cells with fewer than 20 positive or fewer than 20 normal test windows": windows are counted **pooled across all machines** in the comparison cell (dataset × anomaly type), not per machine. PR-AUC is computed per machine, averaged over the machines with at least one positive window in that cell (machines with none are skipped), and the cluster bootstrap resamples those machines | The prereg did not say whether a cell is a machine or a comparison. Made after SMD data access but before any detector was run. Window counts were known (machine-1-5 has 16 positives, machine-2-8 has 17); no scores or metrics were | All verdicts in E2, E3, E5 (and E4 with scenarios as clusters). Implemented in `src/bootstrap.py` (`machines_with_positives`, `pooled_window_counts`, `cell_has_enough_windows`) |
| 2026-11-19 | **Clarification (before any results)** of prereg §4 F2 "PCA reconstruction error, from PCA fitted on training data": the number of PCA components is the smallest that explains **95% of the training variance** (scaler and PCA fitted on the training points only) | The prereg did not fix the number of components. Fixed before any detector was run; not tuned (SMD gives 8–20 components per machine) | F2, so every set containing F2 (F1+F2, F1+F2+F3) in E2–E5. Implemented in `src/features_simple.py` (`PCA_VARIANCE_KEPT = 0.95`) |
