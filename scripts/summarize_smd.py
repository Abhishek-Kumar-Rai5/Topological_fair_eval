"""Summarise an SMD run (results/<exp>/<exp>_results.csv) under prereg §6 and §8.

    python scripts/summarize_smd.py results/E3 [--seed 0]

Per feature set (prereg §8 + deviations_log.md clarification):
  1. average each metric over the 5 seeds, per machine
  2. PR-AUC and recall: only machines with at least one positive test window (others have no
     defined value); false alarms per day: all machines
  3. mean over machines with a 95% cluster bootstrap CI (2,000 resamples, resampling machines)
Comparisons Δ (F1+F2+F3 − F1+F2) and Δs (F1+F3 − F1+F2) on PR-AUC, same machines, paired:
  Helps          Δ > 0 and the CI excludes 0
  No added value the CI lies inside ±0.02
  Inconclusive   anything else, or fewer than 20 positive / 20 normal test windows (pooled)
If both "Helps" and "No added value" hold, the table order is followed (Helps) and the case is flagged.
Writes <exp>_summary.csv and <exp>_verdicts.csv (+ .meta.json) next to the results.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bootstrap import (average_over_seeds, cell_has_enough_windows, cluster_bootstrap_ci,
                           machines_with_positives, paired_delta_ci)
from src.config import load_config
from src.metadata import run_metadata, save_result

MARGIN = 0.02
BOOT = dict(n_resamples=2000, ci=0.95)


def summarize(df, seed):
    pos_machines = machines_with_positives(df)
    rows = []
    for fs, g in df.groupby("feature_set", sort=False):
        row = {"feature_set": fs, "n_machines_with_positives": len(pos_machines),
               "n_machines": g["machine"].nunique()}
        for metric, only_pos in (("pr_auc", True), ("recall_at_thr", True), ("false_alarms_per_day", False)):
            per_machine = average_over_seeds(g, "machine", metric)
            if only_pos:
                per_machine = per_machine.loc[pos_machines]
            mean, lo, hi = cluster_bootstrap_ci(per_machine.to_numpy(), seed=seed, **BOOT)
            row.update({metric: mean, f"{metric}_lo": lo, f"{metric}_hi": hi})
        row["runtime_per_window_ms_mean"] = g["runtime_per_window_ms"].mean()
        row["runtime_per_window_ms_median"] = g["runtime_per_window_ms"].median()
        row["val_alarm_rate_mean"] = g["val_alarm_rate"].mean()
        rows.append(row)
    return pd.DataFrame(rows), pos_machines


def verdict(df, a, b, pos_machines, seed):
    pa = average_over_seeds(df[df.feature_set == a], "machine", "pr_auc").loc[pos_machines]
    pb = average_over_seeds(df[df.feature_set == b], "machine", "pr_auc").loc[pos_machines]
    delta, lo, hi = paired_delta_ci(pa, pb, seed=seed, **BOOT)
    cell = df[(df.feature_set == b) & df.machine.isin(pos_machines)]
    enough = cell_has_enough_windows(df[df.feature_set == b])
    helps = delta > 0 and lo > 0
    none = -MARGIN <= lo and hi <= MARGIN
    if not enough:
        v = "Inconclusive (fewer than 20 positive or normal windows)"
    elif helps:
        v = "Helps" + (" (CI also inside ±0.02)" if none else "")
    elif none:
        v = "No added value"
    else:
        v = "Inconclusive"
    n_pos = int(df[df.feature_set == b].groupby("machine")["n_positive"].first().sum())
    n_norm = int(df[df.feature_set == b].groupby("machine")["n_normal"].first().sum())
    return {"comparison": f"{a} - {b}", "delta_pr_auc": delta, "ci_lo": lo, "ci_hi": hi, "verdict": v,
            "pooled_positive_windows": n_pos, "pooled_normal_windows": n_norm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--seed", type=int, default=0, help="seed of the bootstrap resampling")
    args = ap.parse_args()
    folder = Path(args.folder)
    exp = folder.name
    df = pd.read_csv(folder / f"{exp}_results.csv")
    cfg = load_config(Path("configs") / {"E2": "e2_fair_retest.yaml", "E3": "e3_swap.yaml"}[exp])
    summary, pos_machines = summarize(df, args.seed)
    sets = set(summary.feature_set)
    pairs = [("F1+F2+F3", "F1+F2"), ("F1+F3", "F1+F2")] if exp == "E3" else [("F1+F2+F3", "F1+F2")]
    verdicts = pd.DataFrame([verdict(df, a, b, pos_machines, args.seed) for a, b in pairs if {a, b} <= sets])
    meta = run_metadata(seed={"model_seeds": cfg["seeds"], "bootstrap_seed": args.seed}, config=cfg,
                        extra={"source": f"{exp}_results.csv", "bootstrap": BOOT})
    save_result(summary, folder / f"{exp}_summary.csv", meta)
    save_result(verdicts, folder / f"{exp}_verdicts.csv", meta)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\n{exp}: {df.machine.nunique()} machines, {len(pos_machines)} with positives\n")
    t = summary.copy()
    for m in ("pr_auc", "recall_at_thr", "false_alarms_per_day"):
        t[m] = [f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]" for _, r in summary.iterrows()]
    print(t[["feature_set", "pr_auc", "recall_at_thr", "false_alarms_per_day",
             "runtime_per_window_ms_mean", "runtime_per_window_ms_median"]].round(3).to_string(index=False))
    print()
    print(verdicts.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
