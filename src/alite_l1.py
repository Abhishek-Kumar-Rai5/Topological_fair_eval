"""L1: per-window scores of E3 / E2 detectors re-computed with the committed code (no retraining choices), plus extra metrics.

group "ours"     F1, F1+F2, F1+F3, F1+F2+F3, F5 (configs/e3_swap.yaml) + RANDOM
group "methods"  TADA, PHANTOM (configs/e2_tada_phantom.yaml)
group "tame"     TAME-IF (configs/e2_tame.yaml, run in .venv-tame)
Each machine file keeps the original columns (pr_auc, roc_auc, ... exactly as the original runner) and adds
f1_at_thr, accuracy_at_thr, best_f1_test_thr, best_f1_point_adjusted. scripts/summarize_l1.py checks pr_auc against the
original results files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.alite import extra_metrics, random_scores
from src.config import load_config
from src.metadata import run_metadata, save_result
from src.metrics import evaluate
from src.threshold import threshold_at_fpr

EXTRA = ["f1_at_thr", "accuracy_at_thr", "best_f1_test_thr", "best_f1_point_adjusted"]


def machine_rows(l1_cfg, group, machine):
    src = load_config(l1_cfg["sources"][group])
    n_thr, store, seen = l1_cfg["n_thresholds"], {}, {}

    def hook(name, seed, scores, labels, thr, val_scores):
        store[(name, seed)] = extra_metrics(scores, labels, thr, n_thr)
        seen.update(labels=labels, n_val=len(val_scores), n_test=len(scores))

    if group == "ours":
        from src import runner
        df = runner.run_machine(src, machine, on_scores=hook)
    else:
        from src import e2_methods
        df, _ = e2_methods.run_machine(src, machine, on_scores=hook)
        if group == "tame":
            df = df[df.feature_set == "TAME-IF"].reset_index(drop=True)
    extra = pd.DataFrame([store[(r.feature_set, r.seed)] for r in df.itertuples()], index=df.index)
    df = pd.concat([df, extra], axis=1)
    df["experiment"] = "L1"
    if group == "ours":                                           # RANDOM: uniform scores, seed 0, same threshold rule
        val, test = random_scores(seen["n_val"], seen["n_test"], l1_cfg["random_seed"])
        thr = threshold_at_fpr(val, src["threshold"]["false_alarm_rate"])
        ev = evaluate(test, seen["labels"], thr, 1)
        rnd = {"experiment": "L1", "machine": machine, "seed": 0, "feature_set": "RANDOM", "pr_auc": ev["pr_auc"],
               "roc_auc": ev["roc_auc"], "n_positive": ev["n_positive"], "n_normal": ev["n_normal"],
               **extra_metrics(test, seen["labels"], thr, n_thr)}
        df = pd.concat([df, pd.DataFrame([rnd])], ignore_index=True)
    return df


def run_machine_and_save(cfg, machine, out_dir, group):
    df = machine_rows(cfg, group, machine)
    meta = run_metadata(seed=cfg["seeds"], config=cfg, extra={"machine": machine, "group": group})
    save_result(df, Path(out_dir) / group / "by_machine" / f"{machine}.csv", meta)
    return df


def merge_group(cfg, machines, out_dir, group):
    folder = Path(out_dir) / group
    df = pd.concat([pd.read_csv(folder / "by_machine" / f"{m}.csv") for m in machines], ignore_index=True)
    meta = run_metadata(seed=cfg["seeds"], config=cfg, extra={"machines": list(machines), "group": group})
    return save_result(df, folder / f"L1_{group}_results.csv", meta)
