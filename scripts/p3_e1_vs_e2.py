"""P3 (prereg §8): fixed TAME-IF on its own windows (E2) minus TAME as released (E1), CI over machines.

    taskset -c 0-3 .venv-tame/bin/python scripts/p3_e1_vs_e2.py

E1 side: TAME's own, unchanged functions `run_isolation_forest` and `run_gmm` (utils/experiment_core.py) on E1's
cached features (external/TAME_e1/cache), same windows, same splits, same seeds (42, 0, 1, 2, 3). The notebook
kept only pooled scores, so this recomputes the per-window scores to get per-machine numbers; it must reproduce
E1's pooled means (checked below). E1's "best detector" is chosen the way TAME's consolidate.py does it: highest
mean test AUROC (here GMM). E2 side: results/E2_tame_own/tame_own_by_machine.csv (fixed IF, own windows).
Per machine (windows lying inside one machine's test span): mean over seeds, then paired difference E2 - E1,
95% cluster bootstrap over machines (2,000 resamples).
Rule (§8): Supported = E2 lower than E1 with the CI excluding 0; Refuted = difference inside ±0.02 (CI inside
±0.02); otherwise Inconclusive. Both ROC-AUC and PR-AUC are tested; the CI of a threshold-free metric does not
depend on the threshold, so E2's threshold change cannot move them.
"""
import os

# E1 ran multi-threaded (taskset 0-3, default BLAS threading). The BLAS thread count changes the PCA output in
# the last digits and moves TAME's Isolation Forest AUROC by about 0.001, so 4 threads are needed to reproduce
# E1's numbers exactly. (The E2 side, run_tame_own_windows.py, used 1 thread; that difference is part of the result.)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "4"

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "TAME"))      # TAME's code (same commit as the E1 copy)

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.bootstrap import average_over_seeds, paired_delta_ci
from src.metadata import run_metadata, save_result

MARGIN = 0.02


def verdict(delta, lo, hi):
    if delta < 0 and hi < 0:
        return "Supported (E2 lower than E1, CI excludes 0)"
    if -MARGIN <= lo and hi <= MARGIN:
        return "Refuted (CI inside ±0.02)"
    return "Inconclusive"


def main():
    import run_tame_own_windows as own
    import utils.experiment_core as EC
    from utils.data_loaders import load_SMD
    from utils.windowing import create_windows_df, window_labels_from_point_labels, window_train_len

    df, y, cols, train_len = load_SMD("data")
    X = np.load(ROOT / "external" / "TAME_e1" / "cache" / "SMD_feats_w200_m3_t1_pi40.npy")
    wl = window_labels_from_point_labels(y, EC.WINDOW, EC.STEP)
    n = min(len(X), len(wl)); X, wl = X[:n], wl[:n]
    tr, va, te = EC.make_splits(n, train_len)
    ytr, yval, yte = wl[tr], wl[va], wl[te]
    normal_train = tr[ytr == 0]
    expected_rate = float(yte.mean()) if yte.mean() > 0 else 0.1
    sc, pca, k, tf = EC.fit_reducer(X, normal_train)
    Ztr_n, Ztr_all, Zval, Zte = tf(normal_train), tf(tr), tf(va), tf(te)

    names, bounds = own.test_spans("data")
    start = EC.STEP * te - train_len
    machine = np.full(len(te), -1)
    for m in range(len(names)):
        machine[(start >= bounds[m]) & (start + EC.WINDOW <= bounds[m + 1])] = m

    rows, pooled = [], {"IsolationForest": [], "GMM": []}
    for seed in EC.SEEDS:
        for det, fn in (("IsolationForest", lambda s: EC.run_isolation_forest(Ztr_all, Zval, Zte, yval, yte, expected_rate, s)),
                        ("GMM", lambda s: EC.run_gmm(Ztr_n, Ztr_all, Zval, Zte, yval, yte, expected_rate, s))):
            metrics, scores = fn(seed)
            pooled[det].append(metrics["auroc"])
            for m, name in enumerate(names):
                sel = machine == m
                if yte[sel].sum() == 0 or (yte[sel] == 0).sum() == 0:
                    continue
                rows.append({"detector": det, "machine": name, "seed": seed,
                             "roc_auc": roc_auc_score(yte[sel], scores[sel]),
                             "pr_auc": average_precision_score(yte[sel], scores[sel])})
    e1 = pd.DataFrame(rows)
    shipped = json.load(open(ROOT / "results" / "E1" / "SMD_results_rerun.json"))["metrics"]["unsupervised"]
    for det in pooled:                                    # E1 reproduced? (pooled AUROC means vs the E1 results file)
        ref = shipped[det]["auroc"]["mean"]
        assert abs(np.mean(pooled[det]) - ref) < 1e-6, (det, np.mean(pooled[det]), ref)
    best = max(pooled, key=lambda d: np.mean(pooled[d]))            # TAME's consolidate.py rule
    e2 = pd.read_csv(ROOT / "results" / "E2_tame_own" / "tame_own_by_machine.csv")

    out = []
    for label, det in (("E1 best detector (" + best + ")", best), ("E1 Isolation Forest", "IsolationForest")):
        for metric in ("roc_auc", "pr_auc"):
            a = average_over_seeds(e2, "machine", metric)
            b = average_over_seeds(e1[e1.detector == det], "machine", metric)
            common = sorted(set(a.index) & set(b.index))
            d, lo, hi = paired_delta_ci(a.loc[common], b.loc[common], n_resamples=2000, ci=0.95, seed=0)
            out.append({"comparison": f"E2 fixed TAME-IF (own windows) - {label}", "metric": metric,
                        "e2_mean": a.loc[common].mean(), "e1_mean": b.loc[common].mean(), "delta": d,
                        "ci_lo": lo, "ci_hi": hi, "n_machines": len(common), "verdict": verdict(d, lo, hi)})
    res = pd.DataFrame(out)
    meta = run_metadata(seed={"model_seeds": list(EC.SEEDS), "bootstrap_seed": 0}, config={"design": __doc__},
                        extra={"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "e1_pooled_auroc_reproduced": {d: float(np.mean(v)) for d, v in pooled.items()},
                               "best_detector_by_mean_auroc": best})
    save_result(res, ROOT / "results" / "P3" / "P3_e1_vs_e2.csv", meta)
    save_result(e1, ROOT / "results" / "P3" / "E1_by_machine.csv", meta)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
