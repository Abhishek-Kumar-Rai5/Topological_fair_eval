# Self re-check: columns of `blank_sheet.csv`

Score each paper blind, from the paper itself. Do not open `tda_records_master.csv`.

Score each item Y (yes) / P (partial) / N (no) / U (unknown: not reported or not readable) / NA (not applicable).

| Code | Criterion |
|---|---|
| R1 | Matched non-topological baseline: same data, split, learner and tuning |
| R2 | Replacement ablation: topology swapped for cheap features, not merely removed |
| R3 | Leakage-safe split: time- or entity-separated; preprocessing and thresholds fitted on training data only |
| R4 | Threshold chosen without test labels or test prevalence |
| R5 | Prevalence-aware metrics: PR-AUC, precision at a fixed recall, or false alarms per unit time |
| R6 | Uncertainty measured across independent units (hosts, days, scenarios), not only across seeds |
| R7 | Cost reported; for "real-time" claims, latency at a stated event rate |
| R8 | Robustness tests: noise, drift or missing data |
| R9 | Adversary who knows the method |
| R10 | Known benchmark flaws acknowledged, or corrected data used |
| R11 | Interpretation claims supported by evidence |
| R12 | Code released |
| R13 | Data available |

OUT — outcome vs the best non-topological comparator in the same paper: Helped / Tied / Lost / Mixed / Untested.
"Tied" = within the reported noise, or less than 1 point apart with no statistical test.

ISO — how the paper isolates topology: Replacement (topology swapped for cheap features) / Removal-only (topology just removed) / Not-isolated.

Write the first word only in OUT and ISO. Fill all 15 columns of `blank_sheet.csv`, save as `literature/recheck/filled_sheet.csv`, then run
`python scripts/agreement.py literature/recheck/filled_sheet.csv`.
