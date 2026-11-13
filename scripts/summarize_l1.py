"""L1 table: PR-AUC, ROC-AUC, F1 and accuracy at our threshold, best-F1 on test labels, point-adjusted best-F1.

    python scripts/summarize_l1.py [--seed 0]
Per row (feature set / method): mean over the machines with positive windows of the per-machine seed mean, with a 95%
cluster bootstrap CI (2,000 resamples). Also checks pr_auc against the original results files (E3, E2m, E2m_tame).
Writes results/L1/L1_table.csv and results/L1/L1_check.csv (+ .meta.json). Exploratory, no verdicts.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bootstrap import average_over_seeds, cluster_bootstrap_ci, machines_with_positives
from src.config import load_config
from src.metadata import run_metadata, save_result

COLS = ["pr_auc", "roc_auc", "f1_at_thr", "accuracy_at_thr", "best_f1_test_thr", "best_f1_point_adjusted"]
ORDER = ["F1", "F1+F2", "F1+F3", "F1+F2+F3", "F5", "TADA", "PHANTOM", "TAME-IF", "RANDOM"]
BOOT = dict(n_resamples=2000, ci=0.95)


def check_against_original(df, ref_path, key=("machine", "seed", "feature_set")):
    ref = pd.read_csv(ref_path)
    m = df.merge(ref[[*key, "pr_auc"]], on=list(key), suffixes=("", "_orig"))
    diff = (m["pr_auc"] - m["pr_auc_orig"]).abs()
    return {"rows_compared": len(m), "rows_in_L1": int((df.feature_set != "RANDOM").sum()),
            "max_abs_diff_pr_auc": float(diff.max()), "rows_differing_gt_1e-9": int((diff > 1e-9).sum())}


def table(df, seed):
    pos = machines_with_positives(df)
    rows = []
    for fs in [f for f in ORDER if f in set(df.feature_set)]:
        g = df[df.feature_set == fs]
        row = {"method": fs, "n_machines": len(pos)}
        for c in COLS:
            per = average_over_seeds(g, "machine", c).loc[pos]
            mean, lo, hi = cluster_bootstrap_ci(per.to_numpy(), seed=seed, **BOOT)
            row.update({c: mean, f"{c}_lo": lo, f"{c}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config("configs/l1_metrics.yaml")
    parts, checks = [], []
    for group, ref in cfg["reference_results"].items():
        path = Path("results/L1") / group / f"L1_{group}_results.csv"
        if not path.exists():
            print(f"missing {path}")
            continue
        d = pd.read_csv(path)
        parts.append(d)
        checks.append({"group": group, "reference": ref, **check_against_original(d, ref)})
    df = pd.concat(parts, ignore_index=True)
    tab, chk = table(df, args.seed), pd.DataFrame(checks)
    meta = run_metadata(seed={"bootstrap_seed": args.seed}, config=cfg, extra={"exploratory": True, "bootstrap": BOOT})
    save_result(tab, Path("results/L1/L1_table.csv"), meta)
    save_result(chk, Path("results/L1/L1_check.csv"), meta)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    out = pd.DataFrame({"method": tab.method})
    for c in COLS:
        out[c] = [f"{r[c]:.3f} [{r[c + '_lo']:.3f}, {r[c + '_hi']:.3f}]" for _, r in tab.iterrows()]
    print(f"L1 (exploratory): mean over {tab.n_machines.iloc[0]} machines with positives, 95% CI\n")
    print(out.to_string(index=False))
    print("\npr_auc against the original results files:")
    print(chk.to_string(index=False))


if __name__ == "__main__":
    main()
