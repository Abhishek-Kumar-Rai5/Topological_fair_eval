"""Summarise E3x (EXPLORATORY; results/E3x/E3x_results.csv): F3 split into H0-only and H1-only on SMD. No prereg verdicts.

    python scripts/summarize_e3x.py results/E3x [--seed 0]

Per feature set: PR-AUC, ROC-AUC, recall at 1% false alarms, false alarms per day, each with a 95% cluster bootstrap CI
over machines (as summarize_smd.py). Paired Δ in PR-AUC with its CI: each F3 variant against F1+F2, and H1-only against
H0-only. The §8 words are shown for orientation only; nothing here decides a prediction.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarize_smd as smd                       # noqa: E402
from src.config import load_config               # noqa: E402
from src.metadata import run_metadata, save_result  # noqa: E402

COMPARISONS = [("F1+F2+F3", "F1+F2"), ("F1+F2+F3-H0", "F1+F2"), ("F1+F2+F3-H1", "F1+F2"), ("F1+F2+F3-H1", "F1+F2+F3-H0")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    folder = Path(args.folder)
    cfg = load_config("configs/e3x_f3_split.yaml")
    df = pd.read_csv(folder / "E3x_results.csv")
    summary, pos = smd.summarize(df, args.seed)
    sets = set(summary.feature_set)
    verdicts = pd.DataFrame([smd.verdict(df, a, b, pos, args.seed) for a, b in COMPARISONS if {a, b} <= sets])
    meta = run_metadata(seed={"model_seeds": cfg["seeds"], "bootstrap_seed": args.seed}, config=cfg,
                        extra={"source": "E3x_results.csv", "exploratory": True, "bootstrap": smd.BOOT})
    save_result(summary, folder / "E3x_summary.csv", meta)
    save_result(verdicts, folder / "E3x_verdicts.csv", meta)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\nE3x (EXPLORATORY): {df.machine.nunique()} machines, {len(pos)} with positives\n")
    t = summary.copy()
    for m in ("pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day"):
        t[m] = [f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]" for _, r in summary.iterrows()]
    print(t[["feature_set", "pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day",
             "runtime_per_window_ms_mean"]].to_string(index=False))
    print("\n", verdicts.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
