"""Run E4 (CTU-13): stage 1 builds the per-scenario tables, stage 2 runs the leave-one-scenario-out folds.

    python scripts/run_e4.py configs/e4_ctu13.yaml [--scenarios 5,7] [--jobs 4] [--resume]
                             [--stage 1|2|both] [--out results/E4]

Stage 1 -> results/E4/by_scenario/<n>.parquet (+ .meta.json): windows and fit-free features. Reads data/ only.
Stage 2 -> results/E4/by_fold/<n>.csv, then results/E4/E4_results.csv. Needs stage 1 of ALL scenarios used.
--resume skips files that are VALID (src.metadata.is_valid_result). Every worker is single-threaded.
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

from src import e4
from src.config import load_config
from src.metadata import is_valid_result


def stage2_job(cfg, scenarios, s, out_dir):
    tables = e4.load_tables(cfg.get("tables_from", out_dir), scenarios)
    e4.run_fold_and_save(cfg, tables, s, out_dir)
    return s


def run_pool(fn, jobs, args_list, label):
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(fn, *a): a for a in args_list}
        for fut in as_completed(futs):
            fut.result()                                   # re-raises worker errors
            done += 1
            print(f"  [{done}/{len(args_list)}] {label} {futs[fut][-2] if label == 'fold' else futs[fut][1]} done ({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--scenarios", help="comma-separated subset (default: all usable)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    ap.add_argument("--out", default="results/E4")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    scenarios = [int(x) for x in args.scenarios.split(",")] if args.scenarios else e4.usable_scenarios(cfg)
    print(f"E4: scenarios {scenarios}, jobs={args.jobs}", flush=True)

    if args.stage in ("1", "both"):
        todo = [s for s in scenarios if not (args.resume and is_valid_result(out / "by_scenario" / f"{s}.parquet"))]
        print(f"stage 1: {len(todo)} of {len(scenarios)} scenarios to build", flush=True)
        run_pool(e4.build_scenario, args.jobs, [(cfg, s, out) for s in todo], "scenario")
    if args.stage in ("2", "both"):
        tables = e4.load_tables(cfg.get("tables_from", out), scenarios)   # E4x reuses the stage-1 tables of E4
        folds = [s for s in scenarios if (tables[s]["label"] == 1).any() and (tables[s]["label"] == 0).any()]
        skipped = sorted(set(scenarios) - set(folds))
        if skipped:
            print(f"stage 2: scenarios without positive and normal windows are not test folds: {skipped}", flush=True)
        todo = [s for s in folds if not (args.resume and is_valid_result(out / "by_fold" / f"{s}.csv"))]
        print(f"stage 2: {len(todo)} of {len(folds)} folds to run", flush=True)
        run_pool(stage2_job, args.jobs, [(cfg, scenarios, s, out) for s in todo], "fold")
        path, _ = e4.merge_results(cfg, folds, out)
        print("wrote", path)


if __name__ == "__main__":
    main()
