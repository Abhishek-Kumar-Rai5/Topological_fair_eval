"""Run one experiment config on SMD with our Isolation Forest pipelines.

    python scripts/run_experiment.py configs/e3_swap.yaml [--machines machine-1-1,machine-1-2]
                                     [--jobs 4] [--resume] [--out results/E3]

Writes results/<experiment>/by_machine/<machine>.csv (+ .meta.json) for every machine, then
results/<experiment>/<experiment>_results.csv (+ .meta.json): one row per machine x seed x feature set.
Every file has a sidecar with git commit, seed(s), config and library versions.
--resume skips machines whose file already exists (use it after a crash or shutdown, with the
same code and config). Machines are independent, so --jobs N runs N of them in parallel;
each worker is single-threaded.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")                    # single CPU thread (prereg §6 runtime)

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.runner import merge_results, pipeline_spec, run_machine_and_save
from src.smd import list_machines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--machines", help="comma-separated subset (default: all 28)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", help="output folder (default results/<experiment>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipeline_spec(cfg)                                  # fail early if this runner can't do the config
    out_dir = Path(args.out or f"results/{cfg['experiment']}")
    machines = args.machines.split(",") if args.machines else list_machines(cfg["data_root"])
    todo = [m for m in machines
            if not (args.resume and (out_dir / "by_machine" / f"{m}.csv").exists())]
    print(f"{cfg['experiment']}: {len(machines)} machines, {len(todo)} to run, jobs={args.jobs}", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_machine_and_save, cfg, m, out_dir): m for m in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()                                # re-raises any error from the worker
            print(f"  [{i}/{len(todo)}] {futures[fut]} done ({time.time() - t0:.0f} s)", flush=True)
    path, _ = merge_results(cfg, machines, out_dir)
    print("wrote", path)


if __name__ == "__main__":
    main()
