"""Run one experiment config on SMD with our Isolation Forest pipelines.

    python scripts/run_experiment.py configs/e3_swap.yaml [--machines machine-1-1,machine-1-2]
                                     [--jobs 4] [--resume] [--out results/E3]

Writes results/<experiment>/by_machine/<machine>.csv (+ .meta.json) for every machine, then
results/<experiment>/<experiment>_results.csv (+ .meta.json): one row per machine x seed x feature set.
Every file has a sidecar with git commit, seed(s), config and library versions.
--resume skips machines whose result file is VALID (non-empty, parses, sidecar present and consistent;
see src.metadata.is_valid_result), so a file cut short by a crash is re-run. Use it with the same code and config. Machines are independent, so --jobs N runs N of them in parallel;
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
from src.metadata import is_valid_result
from src.runner import merge_results, pipeline_spec, run_machine_and_save
from src.smd import list_machines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--machines", help="comma-separated subset (default: all 28)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", help="output folder (default results/<experiment>)")
    ap.add_argument("--methods", action="store_true",
                    help="E2 only: run the re-tested methods TADA, PHANTOM, TAME (src/e2_methods.py) instead of our "
                         "pipelines; results go to results/E2m/E2m_results.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.methods:                                    # E2 re-tested methods (own runner, own result name "E2m")
        assert cfg["experiment"] == "E2", "--methods is for configs/e2_fair_retest.yaml"
        from src import e2_methods
        cfg["experiment"] = "E2m"
        run_one, merge = e2_methods.run_machine_and_save, e2_methods.merge_results
    elif cfg["experiment"] == "E5":                       # injections: own runner (src/e5.py)
        from src import e5
        run_one, merge = e5.run_machine_and_save, e5.merge_results
    elif cfg["experiment"] == "E5b":                      # exploratory: k channels per injection (src/e5b.py)
        from src import e5b
        run_one, merge = e5b.run_machine_and_save, e5b.merge_results
    else:
        pipeline_spec(cfg)                              # fail early if this runner can't do the config
        run_one, merge = run_machine_and_save, merge_results
    out_dir = Path(args.out or f"results/{cfg['experiment']}")
    machines = args.machines.split(",") if args.machines else list_machines(cfg["data_root"])
    todo = [m for m in machines
            if not (args.resume and is_valid_result(out_dir / "by_machine" / f"{m}.csv"))]
    print(f"{cfg['experiment']}: {len(machines)} machines, {len(todo)} to run, jobs={args.jobs}", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, cfg, m, out_dir): m for m in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()                                # re-raises any error from the worker
            print(f"  [{i}/{len(todo)}] {futures[fut]} done ({time.time() - t0:.0f} s)", flush=True)
    path, _ = merge(cfg, machines, out_dir)
    print("wrote", path)


if __name__ == "__main__":
    main()
