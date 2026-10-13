"""Summarise E5b (exploratory; results/E5b/E5b_results.csv). Nothing here decides P1 or P2.

    python scripts/summarize_e5b.py results/E5b [--seed 0]

Same cells and statistics as summarize_e5.py (PR-AUC, ROC-AUC, recall, false alarms per day with 95% cluster
bootstrap CIs over machines; paired Δ = PR-AUC(F1+F2+F3) - PR-AUC(F1+F2) and its §8 verdict), plus one rule fixed
before the run: a cell COUNTS only if the F1+F2 baseline ROC-AUC (mean over machines) is at least 0.7. Cells
below that are listed with their numbers and marked "Baseline fails"; they get no verdict.
Writes E5b_summary.csv, E5b_verdicts.csv, E5b_curves.png (+ metadata) and prints the placement report
(machines skipped for a type because fewer than k = 5 channels were usable).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarize_e5 as e5s                       # noqa: E402
from src.config import load_config               # noqa: E402
from src.metadata import run_metadata, save_result  # noqa: E402

BASE = "F1+F2"
FAILS = "Baseline fails (ROC-AUC below the minimum): cell not counted"


def apply_baseline_rule(summary, verdicts, min_roc):
    """Add baseline numbers and a 'counts' flag to the verdicts; replace the verdict of failing cells."""
    base = summary[summary.feature_set == BASE].set_index(["type", "magnitude"])
    v = verdicts.copy()
    key = list(zip(v["type"], v["magnitude"]))
    v["baseline_roc_auc"] = [base.loc[k, "roc_auc"] for k in key]
    v["baseline_pr_auc"] = [base.loc[k, "pr_auc"] for k in key]
    v["counts"] = v["baseline_roc_auc"] >= min_roc
    v["verdict_if_counted"] = v["verdict"]
    v.loc[~v["counts"], "verdict"] = FAILS
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    folder = Path(args.folder)
    cfg = load_config("configs/e5b_injections.yaml")
    df = pd.read_csv(folder / "E5b_results.csv")
    summary, verdicts = e5s.summarize_cells(df, args.seed)
    verdicts = apply_baseline_rule(summary, verdicts, cfg["min_baseline_roc_auc"])
    meta = run_metadata(seed={"model_seeds": cfg["seeds"], "bootstrap_seed": args.seed}, config=cfg,
                        extra={"source": "E5b_results.csv", "exploratory": True})
    save_result(summary, folder / "E5b_summary.csv", meta)
    save_result(verdicts, folder / "E5b_verdicts.csv", meta)
    e5s.plot_curves(summary, folder / "E5b_curves.png")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 70)
    base_rate = (df[df.feature_set == BASE].groupby(["type", "magnitude", "machine"])[["n_positive", "n_normal"]]
                 .first().groupby(["type", "magnitude"]).sum())
    base_rate = (base_rate.n_positive / (base_rate.n_positive + base_rate.n_normal)).rename("base_rate")
    t = verdicts.merge(base_rate.reset_index(), on=["type", "magnitude"])
    print(f"\nE5b (EXPLORATORY): {df.machine.nunique()} machines, minimum baseline ROC-AUC {cfg['min_baseline_roc_auc']}\n")
    print(t[["type", "magnitude", "is_decision", "base_rate", "baseline_pr_auc", "baseline_roc_auc", "counts",
             "delta_pr_auc", "ci_lo", "ci_hi", "verdict"]].round(4).to_string(index=False))
    print(f"\ncells that count: {int(verdicts.counts.sum())} of {len(verdicts)}")
    place = pd.read_csv(folder / "E5b_placement.csv")
    print(f"\nPLACEMENT: {int(place.skipped.sum())} of {len(place)} machine x type cells skipped (fewer than k usable channels)")
    print(place[place.skipped].groupby("type").size().rename("machines_skipped").to_string())
    short = place[(~place.skipped) & (place.n_placed < place.n_requested)]
    print(f"{len(short)} further cells placed fewer than {int(place.n_requested.iloc[0])} injections")


if __name__ == "__main__":
    main()
