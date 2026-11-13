"""L4: MST edge lengths against ripser H0 deaths on the F3 clouds of one machine, and the time of each (EXPLORATORY).

    python scripts/l4_mst.py [--config configs/l4_mst.yaml]
For every test window of the machine: the F3 cloud (z-scored with training mean/std, m=3, tau=1), its distance matrix,
then (a) sorted MST edge lengths, (b) sorted finite H0 death times of ripser (maxdim 0). Reports the largest absolute
difference over all windows and the mean time per window of each path (pdist+squareform included in both; also alone).
Single thread, best of `repeats` passes. Writes results/L4/L4_mst.csv (+ .meta.json).
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
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

from src.alite import h0_deaths_ripser, mst_h0_deaths
from src.config import load_config
from src.features_topo import delay_embed, zscore_params
from src.metadata import run_metadata, save_result
from src.prepare import prepare_machine


def best_per_window(fn, items, repeats):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        for x in items:
            fn(x)
        best = min(best, time.perf_counter() - t0)
    return 1000 * best / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/l4_mst.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    w = cfg["windows"]
    mw = prepare_machine(cfg["machine"], cfg["data_root"], w["size"], w["step"], w["label_tail"], cfg["split"]["val_frac"])
    mean, std = zscore_params(mw.train)
    clouds = [delay_embed((win - mean) / std) for win in mw.test]
    diffs, count_mismatch = [], 0
    for c in clouds:
        d = squareform(pdist(c))
        m, h = mst_h0_deaths(d), h0_deaths_ripser(d)
        if len(m) != len(h):
            count_mismatch += 1
        else:
            diffs.append(float(np.abs(m - h).max()) if len(m) else 0.0)
    r = cfg["repeats"]
    t_dist = best_per_window(lambda c: squareform(pdist(c)), clouds, r)
    t_mst = best_per_window(lambda c: mst_h0_deaths(squareform(pdist(c))), clouds, r)
    t_rip = best_per_window(lambda c: ripser(squareform(pdist(c)), maxdim=0, distance_matrix=True), clouds, r)
    row = {"machine": cfg["machine"], "n_windows": len(clouds), "points_per_window": len(clouds[0]),
           "max_abs_diff": max(diffs), "median_abs_diff": float(np.median(diffs)), "windows_count_mismatch": count_mismatch,
           "ms_per_window_distances_only": t_dist, "ms_per_window_mst_path": t_mst, "ms_per_window_ripser_h0_path": t_rip,
           "ripser_over_mst": t_rip / t_mst}
    df = pd.DataFrame([row])
    save_result(df, Path("results/L4/L4_mst.csv"), run_metadata(seed=None, config=cfg, extra={"exploratory": True}))
    print(df.T.to_string(header=False))


if __name__ == "__main__":
    main()
