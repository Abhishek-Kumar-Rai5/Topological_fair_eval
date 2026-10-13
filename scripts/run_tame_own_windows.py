"""Fixed TAME on TAME's OWN windows (for P3, prereg §5 and §7): run in .venv-tame.

    .venv-tame/bin/python scripts/run_tame_own_windows.py [--jobs 2] [--out results/E2_tame_own]

TAME's released SMD protocol is kept (utils/data_loaders.py, utils/experiment_core.py @ e77c0d9):
    all 28 machines are concatenated into ONE stream (train, then test), windows of 200 with step 200,
    window label = 1 if any point in it is anomalous, official train/test boundary, the last 18% of the
    train windows are validation, StandardScaler + PCA(50) fitted on the normal train windows, TAME's own
    persistence-image features (imported from external/TAME, never copied), seeds 42, 0, 1, 2, 3.
Exactly the two prereg §5 changes, nothing else:
    1. detector fixed in advance: Isolation Forest (200 trees, as in TAME). No "best detector on test".
    2. threshold = 1% false alarms on VALIDATION windows (replaces TAME's test-prevalence fallback).
No test information is used before the scores and the threshold exist.

Outputs in --out: tame_own_results.csv (pooled, one row per seed), tame_own_by_machine.csv (windows lying
inside one machine's test span, per machine and seed), each with a .meta.json (commit, seeds, versions,
script hash). Feature chunks are cached in data/tame_cache/ so a shutdown loses at most one chunk.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "TAME"))     # TAME's own `utils` and `models` packages

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.metadata import run_metadata, save_result
from src.metrics import count_alarm_events
from src.threshold import threshold_at_fpr

CHUNK = 250            # windows per cached feature chunk


def extract_features(windows, feature_cols, cache_dir, n_jobs):
    """TAME's extractor on every window; cached per chunk (atomic) so a crash loses little.

    Returns (X, seconds_per_window) where the time is for extraction only (cached chunks keep their time).
    """
    from utils.tda_features import TDAPersistenceImageExtractor
    from utils.experiment_core import M_EMBED, PI_RES, SIGMA, TAU
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = TDAPersistenceImageExtractor(M_EMBED, TAU, PI_RES, SIGMA)
    ext._ensure_gtda()
    ext._vr.n_jobs = n_jobs                              # TAME sets -1 (all cores); we keep the load moderate
    parts, seconds = [], 0.0
    for c, start in enumerate(range(0, len(windows), CHUNK)):
        f, meta = cache_dir / f"chunk_{c:04d}.npy", cache_dir / f"chunk_{c:04d}.json"
        chunk = windows[start:start + CHUNK]
        X = None
        try:                                             # a chunk cut short by a crash is recomputed
            info = json.loads(meta.read_text())
            X, t = np.load(f), info["seconds"]
            if info.get("n") != len(chunk) or len(X) != len(chunk):
                X = None
        except (OSError, ValueError, KeyError, EOFError):
            X = None
        if X is None:
            t0 = time.perf_counter()
            X = ext.transform_windows(chunk, feature_cols, verbose_every=10_000)
            t = time.perf_counter() - t0
            np.save(str(f) + ".tmp.npy", X)
            os.replace(str(f) + ".tmp.npy", f)
            tmp = str(meta) + ".tmp"
            Path(tmp).write_text(json.dumps({"n": len(chunk), "seconds": t}))
            os.replace(tmp, meta)                        # the .json is written last: it marks the chunk as done
            print(f"  features: chunk {c + 1}/{-(-len(windows) // CHUNK)} ({t:.0f} s)", flush=True)
        parts.append(X)
        seconds += t
    return np.vstack(parts), seconds / len(windows)


def test_spans(data_root):
    """Test-stream start of every machine, in the (sorted-file) order TAME's loader uses."""
    files = sorted(Path(data_root, "SMD", "test").glob("*.txt"))
    lengths = [sum(1 for _ in open(f)) for f in files]
    return [f.stem for f in files], np.r_[0, np.cumsum(lengths)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--cache-dir", default="data/tame_cache")
    ap.add_argument("--out", default="results/E2_tame_own")
    ap.add_argument("--jobs", type=int, default=2, help="cores for giotto-tda (TAME itself uses all)")
    args = ap.parse_args()

    from utils.data_loaders import load_SMD
    from utils.experiment_core import SEEDS, WINDOW, STEP, fit_reducer, make_splits
    from utils.windowing import create_windows_df, window_labels_from_point_labels, window_train_len

    df, y, feature_cols, train_len = load_SMD(args.data_root)
    windows = create_windows_df(df, WINDOW, STEP)
    X, feat_s = extract_features(windows, feature_cols, Path(args.cache_dir), args.jobs)
    wl = window_labels_from_point_labels(y, WINDOW, STEP)
    n = min(len(X), len(wl))
    X, wl = X[:n], wl[:n]
    tr, va, te = make_splits(n, train_len)
    ytr, yval, yte = wl[tr], wl[va], wl[te]
    assert yval.sum() == 0 and ytr.sum() == 0, "SMD train/validation windows must be all normal"
    sc, pca, k, tf = fit_reducer(X, tr[ytr == 0])
    Ztr, Zva, Zte = tf(tr), tf(va), tf(te)

    # which test windows lie wholly inside one machine's test span (others cross a boundary, dropped)
    names, bounds = test_spans(args.data_root)
    twl = window_train_len(train_len, WINDOW, STEP)
    start = STEP * (np.arange(len(wl))[te]) - train_len                  # window start in test coordinates
    machine = np.full(len(te), -1)
    for m in range(len(names)):
        machine[(start >= bounds[m]) & (start + WINDOW <= bounds[m + 1])] = m
    test_days = (bounds[-1]) * 1.0 / 1440                                # 1 minute per point (data/README.md)

    pooled, per_machine = [], []
    for seed in SEEDS:
        forest = IsolationForest(n_estimators=200, random_state=seed).fit(Ztr)   # TAME's IF, fixed in advance
        s_va, s_te = -forest.score_samples(Zva), -forest.score_samples(Zte)
        thr = threshold_at_fpr(s_va, 0.01)                                        # validation only
        alarm = s_te > thr
        false_alarm = alarm & (yte == 0)
        pooled.append({"seed": seed, "n_train_windows": len(tr), "n_val_windows": len(va),
                       "n_test_windows": len(te), "n_positive": int(yte.sum()), "n_normal": int((yte == 0).sum()),
                       "threshold": thr, "val_alarm_rate": float((s_va > thr).mean()),
                       "pr_auc": float(average_precision_score(yte, s_te)), "roc_auc": float(roc_auc_score(yte, s_te)),
                       "recall_at_thr": float(alarm[yte == 1].mean()),
                       "false_alarm_events": count_alarm_events(false_alarm),
                       "false_alarms_per_day": count_alarm_events(false_alarm) / test_days,
                       "pca_k": k, "feature_dim": X.shape[1], "feature_s_per_window": feat_s})
        for m, name in enumerate(names):
            sel = machine == m
            ym, sm = yte[sel], s_te[sel]
            if ym.sum() == 0 or (ym == 0).sum() == 0:
                continue                                   # no defined PR-AUC / ROC-AUC on this machine
            per_machine.append({"machine": name, "seed": seed, "n_positive": int(ym.sum()),
                                "n_normal": int((ym == 0).sum()), "pr_auc": float(average_precision_score(ym, sm)),
                                "roc_auc": float(roc_auc_score(ym, sm)), "recall_at_thr": float(alarm[sel][ym == 1].mean())})
    meta = run_metadata(seed=list(SEEDS), config={"protocol": "TAME released SMD protocol + two prereg §5 changes",
                                                  "window": WINDOW, "step": STEP, "detector": "IsolationForest(200)"},
                        extra={"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "tame_commit": "e77c0d93434bdd6b20db9db60e524a30a20a35b5"})
    out = Path(args.out)
    save_result(pd.DataFrame(pooled), out / "tame_own_results.csv", meta)
    save_result(pd.DataFrame(per_machine), out / "tame_own_by_machine.csv", meta)
    p = pd.DataFrame(pooled)
    print(p[["seed", "pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day", "val_alarm_rate"]].round(3).to_string(index=False))
    print(f"test windows {len(te)} ({int(yte.sum())} positive), per-machine rows {len(per_machine)}, feature s/window {feat_s:.3f}")


if __name__ == "__main__":
    main()
