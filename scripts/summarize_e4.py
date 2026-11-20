"""Summarise E4 (results/E4/E4_results.csv) under prereg §6 and §8, cluster = scenario.

    python scripts/summarize_e4.py results/E4 [--seed 0]

Reuses scripts/summarize_smd.py (merged from master): per feature set PR-AUC, ROC-AUC, recall and false alarms per
normal host-day, each with a 95% cluster bootstrap CI over scenarios; and the Helps / No added value / Inconclusive
rule for  Δ = F1+F2+F3 - F1+F2  and  Δz = Z - Z-control.  Also per behaviour type (configs/e4_behaviours.yaml).
Writes E4_summary.csv, E4_verdicts.csv, E4_by_type.csv next to the results (with .meta.json).
Type cells have 2 scenarios each at most (Shape: 8, 12; Volume: 4, 11), so their CIs are very wide.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarize_smd as smd                       # noqa: E402
from src.config import load_config               # noqa: E402
from src.metadata import run_metadata, save_result  # noqa: E402

COMPARISONS = [("F1+F2+F3", "F1+F2"), ("Z", "Z-control")]


def verdict_table(df, seed, extra=None):
    summary, pos = smd.summarize(df, seed)
    sets = set(summary.feature_set)
    rows = [smd.verdict(df, a, b, pos, seed) for a, b in COMPARISONS if {a, b} <= sets]
    v = pd.DataFrame(rows)
    for k, val in (extra or {}).items():
        v.insert(0, k, val)
    return summary, v, pos


def by_type(df, types, seed):
    out = []
    for t, g in df.assign(type=df["scenario"].map(types)).groupby("type"):
        if t == "Unknown":
            continue
        _, v, pos = verdict_table(g, seed, {"type": t, "n_scenarios": g["scenario"].nunique()})
        out.append(v)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--seed", type=int, default=0, help="seed of the bootstrap resampling")
    args = ap.parse_args()
    folder = Path(args.folder)
    df = pd.read_csv(folder / "E4_results.csv")
    cfg = load_config("configs/e4_ctu13.yaml")
    types = {int(k): v["type"] for k, v in yaml.safe_load(Path(cfg["behaviours_file"]).read_text())["scenarios"].items()}
    summary, verdicts, pos = verdict_table(df, args.seed)
    typed = by_type(df, types, args.seed)
    meta = run_metadata(seed={"model_seeds": cfg["seeds"], "bootstrap_seed": args.seed}, config=cfg,
                        extra={"source": "E4_results.csv", "bootstrap": smd.BOOT})
    save_result(summary, folder / "E4_summary.csv", meta)
    save_result(verdicts, folder / "E4_verdicts.csv", meta)
    if len(typed):
        save_result(typed, folder / "E4_by_type.csv", meta)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\nE4: {df.scenario.nunique()} test scenarios, {len(pos)} with positives\n")
    t = summary.copy()
    for m in ("pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day"):
        t[m] = [f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]" for _, r in summary.iterrows()]
    print(t[["feature_set", "pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day"]].to_string(index=False))
    print("\n", verdicts.round(4).to_string(index=False))
    if len(typed):
        print("\nBY BEHAVIOUR TYPE (README fallback; Unknown scenarios are left out):\n", typed.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
