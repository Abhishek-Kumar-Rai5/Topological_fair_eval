"""Summarise L2b (EXPLORATORY; results/L2b/L2b_results.csv): per-channel H0 / H1 with tau from autocorrelation, SMD.

    python scripts/summarize_l2b.py [--seed 0]
Table and Δ in PR-AUC with 95% cluster CIs over machines (as E3), §8 words for orientation. The rule fixed in advance:
H1 "Helps" (F1+F2+PC-H1 vs F1+F2) -> "loops help with a proper embedding"; otherwise "H1 adds nothing even with an
H1-friendly embedding". Also: the tau chosen per machine (training data only) and F1+F2 against E3's F1+F2.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarize_smd as smd                       # noqa: E402
from src.alite import tau_ac                      # noqa: E402
from src.config import load_config               # noqa: E402
from src.metadata import run_metadata, save_result  # noqa: E402
from src.prepare import prepare_machine           # noqa: E402

COMPARISONS = [("F1+F2+PC-H1", "F1+F2"), ("F1+F2+PC-H0", "F1+F2"), ("F1+F2+PC-H1", "F1+F2+PC-H0")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    folder = Path("results/L2b")
    cfg = load_config("configs/l2b_h1_friendly.yaml")
    df = pd.read_csv(folder / "L2b_results.csv")
    summary, pos = smd.summarize(df, args.seed)
    verdicts = pd.DataFrame([smd.verdict(df, a, b, pos, args.seed) for a, b in COMPARISONS])
    w = cfg["windows"]
    taus = pd.DataFrame([{"machine": m, "tau_ac": tau_ac(prepare_machine(m, cfg["data_root"], w["size"], w["step"],
                                                                         w["label_tail"], cfg["split"]["val_frac"]).train)}
                         for m in sorted(df.machine.unique())])
    e3 = pd.read_csv("results/E3/E3_results.csv")
    e3 = e3[e3.feature_set == "F1+F2"].merge(df[df.feature_set == "F1+F2"], on=["machine", "seed"], suffixes=("_e3", ""))
    same = float((e3.pr_auc_e3 - e3.pr_auc).abs().max())
    meta = run_metadata(seed={"model_seeds": cfg["seeds"], "bootstrap_seed": args.seed}, config=cfg,
                        extra={"exploratory": True, "bootstrap": smd.BOOT, "f1f2_max_abs_diff_vs_E3": same})
    save_result(summary, folder / "L2b_summary.csv", meta)
    save_result(verdicts, folder / "L2b_verdicts.csv", meta)
    save_result(taus, folder / "L2b_tau.csv", meta)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"\nL2b (EXPLORATORY): {df.machine.nunique()} machines, {len(pos)} with positives; F1+F2 vs E3 max |ΔPR-AUC| = {same:.2e}\n")
    t = summary.copy()
    for m in ("pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day"):
        t[m] = [f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]" for _, r in summary.iterrows()]
    print(t[["feature_set", "pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day", "runtime_per_window_ms_mean"]]
          .to_string(index=False))
    print("\n", verdicts.round(4).to_string(index=False))
    h1 = verdicts.iloc[0].verdict
    print("\nFixed rule ->", "loops help with a proper embedding" if h1.startswith("Helps")
          else "H1 adds nothing even with an H1-friendly embedding", f"(H1 vs F1+F2: {h1})")
    print("tau_ac per machine:", taus.tau_ac.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
