"""Run our Isolation Forest pipelines on SMD: config -> one result row per (machine, seed, feature set).

Used for E3 (`feature_sets`) and for the "our pipelines" part of E2 (`our_pipelines.feature_sets`).
TADA, PHANTOM, TAME (E2/E1), CTU-13 (E4) and the injections (E5) need their own adapters.

Per machine (prereg §2, §5):
    1. prepare windows: train (first 80%), validation (last 20%), test + labels
    2. feature groups (F1, F2, F3, F5) are fitted on TRAINING windows only, then applied to
       train, validation and test. They do not depend on the seed, so this is done once.
    3. for each seed and feature set: fit the detector on training features, set the
       threshold at 1% false alarms on VALIDATION scores, score the test windows, compute metrics.
The test labels are used only inside `evaluate`, after the scores and threshold exist.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.detector import IFDetector
from src.knn_detector import KNNDetector
from src.feature_sets import GROUPS
from src.metadata import run_metadata, save_result
from src.metrics import evaluate
from src.prepare import prepare_machine
from src.threshold import threshold_at_fpr


def pipeline_spec(cfg):
    """(detector config, feature-set names) of the configs this runner can run."""
    if "our_pipelines" in cfg:                       # E2
        return cfg["our_pipelines"]["detector"], cfg["our_pipelines"]["feature_sets"]
    if "feature_sets" in cfg and "injection" not in cfg:   # E3
        return cfg["detector"], cfg["feature_sets"]
    raise ValueError(f"{cfg.get('experiment')}: this runner does only the Isolation Forest "
                     "pipelines on real SMD (E2 our_pipelines, E3)")


def group_features(mw, groups):
    """Fit each feature group on the training windows; return its features and timings.

    Returns {group: dict(train, val, test, seconds_per_window)}.
    seconds_per_window = time to compute the group's features on the TEST windows / n windows.
    """
    out = {}
    for g in groups:
        fitted = GROUPS[g][0]().fit(mw.train)         # training windows only
        train, val = fitted.transform(mw.train), fitted.transform(mw.val)
        t0 = time.perf_counter()
        test = fitted.transform(mw.test)
        out[g] = {"train": train, "val": val, "test": test,
                  "seconds_per_window": (time.perf_counter() - t0) / len(mw.test)}
    return out


def make_detector(det_cfg, seed):
    """Isolation Forest (E2, E3, ...) or, for the exploratory L3, a kNN distance detector."""
    if det_cfg["name"] == "isolation_forest":
        return IFDetector(seed, det_cfg["n_estimators"], det_cfg.get("max_samples", "auto"))
    if det_cfg["name"] == "knn":
        return KNNDetector(det_cfg.get("k", 5))
    raise ValueError(det_cfg)


def run_machine(cfg, machine, on_scores=None):
    """All result rows (every seed, every feature set) for one machine.

    on_scores(name, seed, scores, labels, threshold, val_scores) is an optional hook (used by L1 to add metrics
    without retraining); it changes nothing in the rows.
    """
    det_cfg, feature_sets = pipeline_spec(cfg)
    w, minutes = cfg["windows"], cfg.get("minutes_per_point", 1.0)
    mw = prepare_machine(machine, root=cfg["data_root"], size=w["size"], step=w["step"],
                         tail=w["label_tail"], val_frac=cfg["split"]["val_frac"])
    groups = sorted({g for name in feature_sets for g in name.split("+")})
    feats = group_features(mw, groups)
    fpr = cfg["threshold"]["false_alarm_rate"]
    rows = []
    for name in feature_sets:
        parts = name.split("+")
        X = {k: np.hstack([feats[p][k] for p in parts]) for k in ("train", "val", "test")}
        feature_seconds = sum(feats[p]["seconds_per_window"] for p in parts)
        for seed in cfg["seeds"]:
            det = make_detector(det_cfg, seed)
            det.fit(X["train"])
            val_scores = det.score(X["val"])
            thr = threshold_at_fpr(val_scores, fpr)                      # validation only
            t0 = time.perf_counter()
            scores = det.score(X["test"])
            score_seconds = (time.perf_counter() - t0) / len(scores)
            row = evaluate(scores, mw.test_labels, thr, mw.n_test_points, minutes)
            if on_scores is not None:
                on_scores(name, seed, scores, mw.test_labels, thr, val_scores)
            rows.append({"experiment": cfg["experiment"], "machine": machine, "seed": seed,
                         "feature_set": name, "n_features": X["train"].shape[1], "threshold": thr,
                         "n_train_windows": len(mw.train), "n_val_windows": len(mw.val),
                         "n_test_windows": len(mw.test),
                         "val_alarm_rate": float((val_scores > thr).mean()),
                         "runtime_per_window_ms": 1000 * (feature_seconds + score_seconds), **row})
    return pd.DataFrame(rows)


def run_machine_and_save(cfg, machine, out_dir):
    """run_machine, then save `<out_dir>/by_machine/<machine>.csv` with its metadata sidecar."""
    df = run_machine(cfg, machine)
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"machine": machine})
    save_result(df, Path(out_dir) / "by_machine" / f"{machine}.csv", meta)
    return df


def merge_results(cfg, machines, out_dir):
    """Join the per-machine files into `<out_dir>/<experiment>_results.csv` (+ metadata)."""
    out_dir = Path(out_dir)
    df = pd.concat([pd.read_csv(out_dir / "by_machine" / f"{m}.csv") for m in machines], ignore_index=True)
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"machines": list(machines)})
    return save_result(df, out_dir / f"{cfg['experiment']}_results.csv", meta)
