"""L2a: synthetic sanity check of H1 and the delay tau (EXPLORATORY, docs/plan_A_lite.md).

    python scripts/run_l2a.py [--config configs/l2a_synthetic.yaml]
For each data seed: 10 channels of sinusoids (periods 24 and 37) + noise, one anomaly (channel 0 at period x2 for 60 samples).
For tau in {1, 6}: the 16 H1 summaries of F3 (whole z-scored window as one cloud), fitted on training windows only.
Reports the maximum H1 persistence (feature max_life) on positive and normal test windows, and the ROC-AUC / PR-AUC of an
Isolation Forest (300 trees, seed = data seed) on those 16 features with the 1%-validation threshold.
Writes results/L2a/L2a_results.csv and L2a_summary.csv (+ .meta.json).
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.alite import WholeWindowH1, synthetic_machine
from src.config import load_config
from src.detector import IFDetector
from src.metadata import run_metadata, save_result
from src.metrics import evaluate
from src.splits import split_train_val
from src.threshold import threshold_at_fpr
from src.windows import label_windows, make_windows

MAX_LIFE = 2          # column of "max_life" in the 16 summaries


def one(cfg, seed, tau):
    a = cfg["anomaly"]
    train, test, point_labels = synthetic_machine(seed, cfg["channels"], cfg["n_train"], cfg["n_test"], cfg["noise"],
                                                  a["start"], a["length"])
    w = cfg["windows"]
    fit_part, val_part = split_train_val(train, cfg["split"]["val_frac"]) if "split" in cfg else split_train_val(train, 0.2)
    tr, _ = make_windows(fit_part, w["size"], w["step"])
    va, _ = make_windows(val_part, w["size"], w["step"])
    te, starts = make_windows(test, w["size"], w["step"])
    labels = label_windows(point_labels, starts, w["size"], w["label_tail"])
    f = WholeWindowH1(tau).fit(tr)
    Xtr, Xva, Xte = f.transform(tr), f.transform(va), f.transform(te)
    det = IFDetector(seed, cfg["detector"]["n_estimators"]).fit(Xtr)
    thr = threshold_at_fpr(det.score(Xva), cfg["threshold"]["false_alarm_rate"])
    ev = evaluate(det.score(Xte), labels, thr, len(test))
    return {"data_seed": seed, "tau": tau, "max_h1_positive_windows": float(Xte[labels == 1, MAX_LIFE].mean()),
            "max_h1_normal_windows": float(Xte[labels == 0, MAX_LIFE].mean()),
            "max_h1_over_all_windows": float(Xte[:, MAX_LIFE].max()),
            "roc_auc": ev["roc_auc"], "pr_auc": ev["pr_auc"], "n_positive": ev["n_positive"], "n_normal": ev["n_normal"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/l2a_synthetic.yaml")
    ap.add_argument("--out", default="results/L2a")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    rows = [one(cfg, s, tau) for s in cfg["data_seeds"] for tau in cfg["taus"]]
    df = pd.DataFrame(rows)
    summ = df.groupby("tau").agg(["mean", "std"]).drop(columns=["data_seed"], level=0)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    meta = run_metadata(seed=cfg["data_seeds"], config=cfg, extra={"exploratory": True, "seconds": time.time() - t0})
    save_result(df, Path(args.out) / "L2a_results.csv", meta)
    save_result(summ, Path(args.out) / "L2a_summary.csv", meta)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.round(3).to_string(index=False))
    print(summ[["tau", "max_h1_positive_windows_mean", "max_h1_normal_windows_mean", "roc_auc_mean", "roc_auc_std",
                "pr_auc_mean"]].round(3).to_string(index=False))
    print(f"runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
