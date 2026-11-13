"""L1: re-score E3 and E2 detectors and add paper-style metrics (EXPLORATORY, docs/plan_A_lite.md).

    python scripts/run_l1.py --group ours|methods [--jobs 4] [--resume]          (.venv)
    .venv-tame/bin/python scripts/run_l1.py --group tame [--jobs 2] [--resume]    (TAME needs giotto-tda)
Writes results/L1/<group>/by_machine/<machine>.csv and results/L1/<group>/L1_<group>_results.csv (+ .meta.json).
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alite_l1 import merge_group, run_machine_and_save
from src.config import load_config
from src.metadata import is_valid_result
from src.smd import list_machines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["ours", "methods", "tame"])
    ap.add_argument("--config", default="configs/l1_metrics.yaml")
    ap.add_argument("--machines")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results/L1")
    args = ap.parse_args()
    cfg = load_config(args.config)
    machines = args.machines.split(",") if args.machines else list_machines(cfg["data_root"])
    todo = [m for m in machines
            if not (args.resume and is_valid_result(Path(args.out) / args.group / "by_machine" / f"{m}.csv"))]
    print(f"L1/{args.group}: {len(machines)} machines, {len(todo)} to run, jobs={args.jobs}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_machine_and_save, cfg, m, args.out, args.group): m for m in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            print(f"  [{i}/{len(todo)}] {futures[fut]} done ({time.time() - t0:.0f} s)", flush=True)
    path, _ = merge_group(cfg, machines, args.out, args.group)
    print("wrote", path, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
