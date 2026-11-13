"""Summarise L3 (EXPLORATORY; results/L3/L3_results.csv): the E3 swap test with a kNN detector, next to E3/E3x Isolation Forest.

    python scripts/summarize_l3.py [--seed 0]
Δ main (F1+F2+F3 - F1+F2), Δs (F1+F3 - F1+F2), ΔH0, ΔH1 in PR-AUC, 95% cluster CIs over machines; §8 words for orientation.
The Isolation Forest values are read from results/E3/E3_verdicts.csv and results/E3x/E3x_verdicts.csv (not recomputed).
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

NAMES = [("Δ main", "F1+F2+F3", "F1+F2"), ("Δs", "F1+F3", "F1+F2"),
         ("ΔH0", "F1+F2+F3-H0", "F1+F2"), ("ΔH1", "F1+F2+F3-H1", "F1+F2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    folder = Path("results/L3")
    cfg = load_config("configs/l3_knn.yaml")
    df = pd.read_csv(folder / "L3_results.csv")
    summary, pos = smd.summarize(df, args.seed)
    iforest = pd.concat([pd.read_csv("results/E3/E3_verdicts.csv"), pd.read_csv("results/E3x/E3x_verdicts.csv")])
    rows = []
    for label, a, b in NAMES:
        k = smd.verdict(df, a, b, pos, args.seed)
        i = iforest[iforest.comparison == f"{a} - {b}"].iloc[0]
        rows.append({"gain": label, "comparison": k["comparison"],
                     "knn_delta": k["delta_pr_auc"], "knn_lo": k["ci_lo"], "knn_hi": k["ci_hi"], "knn_verdict": k["verdict"],
                     "if_delta": i.delta_pr_auc, "if_lo": i.ci_lo, "if_hi": i.ci_hi, "if_verdict": i.verdict})
    out = pd.DataFrame(rows)
    meta = run_metadata(seed={"model_seeds": cfg["seeds"], "bootstrap_seed": args.seed}, config=cfg,
                        extra={"exploratory": True, "bootstrap": smd.BOOT})
    save_result(summary, folder / "L3_summary.csv", meta)
    save_result(out, folder / "L3_verdicts.csv", meta)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\nL3 (EXPLORATORY): kNN k=5, {df.machine.nunique()} machines, {len(pos)} with positives\n")
    t = summary.copy()
    for m in ("pr_auc", "roc_auc"):
        t[m] = [f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]" for _, r in summary.iterrows()]
    print(t[["feature_set", "pr_auc", "roc_auc", "runtime_per_window_ms_mean"]].to_string(index=False))
    show = pd.DataFrame({"gain": out.gain,
                         "kNN Δ [CI]": [f"{r.knn_delta:+.3f} [{r.knn_lo:+.3f}, {r.knn_hi:+.3f}]" for r in out.itertuples()],
                         "kNN word": out.knn_verdict,
                         "IF Δ [CI]": [f"{r.if_delta:+.3f} [{r.if_lo:+.3f}, {r.if_hi:+.3f}]" for r in out.itertuples()],
                         "IF word": out.if_verdict})
    print("\n", show.to_string(index=False))


if __name__ == "__main__":
    main()
