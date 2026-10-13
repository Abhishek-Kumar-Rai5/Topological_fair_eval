"""E2 table: our pipelines, TADA, PHANTOM and TAME on SMD (prereg §5, §6, §7).

    python scripts/summarize_e2.py [--seed 0]

Reads (whatever exists):
    results/E2/E2_results.csv            our F1+F2 and F1+F2+F3        (run_experiment.py)
    results/E2m/E2m_results.csv          TADA, PHANTOM                 (run_experiment.py --methods)
    results/E2m_tame/E2m_results.csv     TAME-IF / -GMM / -OCSVM on OUR windows (.venv-tame)
    results/E2_tame_own/*.csv            TAME-IF on TAME's OWN windows (scripts/run_tame_own_windows.py)
Same method as summarize_smd.py: per machine mean over seeds, machines with a positive window, 95% cluster
bootstrap over machines (2,000 resamples). TAME on its own windows is computed per machine from the windows that
lie inside one machine's test span; its false alarms per day are pooled over the whole stream, so they have no CI.
Writes results/E2m/E2_table.csv (+ .meta.json). No verdicts here: P3 needs E1.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarize_smd as smd                                   # noqa: E402
from src.bootstrap import average_over_seeds, cluster_bootstrap_ci  # noqa: E402
from src.metadata import run_metadata, save_result            # noqa: E402

PARTS = [("our pipelines (our windows)", "results/E2/E2_results.csv"),
         ("TADA (our windows)", "results/E2m/E2m_results.csv"),
         ("TAME (our windows)", "results/E2m_tame/E2m_results.csv")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    frames = []
    for label, path in PARTS:
        if Path(path).exists():
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    summary, pos = smd.summarize(df, args.seed)
    summary.insert(1, "windows", ["our (60/10)"] * len(summary))

    own = Path("results/E2_tame_own")
    if (own / "tame_own_by_machine.csv").exists():
        pm = pd.read_csv(own / "tame_own_by_machine.csv")
        pooled = pd.read_csv(own / "tame_own_results.csv")
        row = {"feature_set": "TAME-IF", "windows": "TAME's own (200/200)",
               "n_machines_with_positives": pm.machine.nunique(), "n_machines": pm.machine.nunique()}
        for m in ("pr_auc", "roc_auc", "recall_at_thr"):
            v = average_over_seeds(pm, "machine", m)
            mean, lo, hi = smd.cluster_bootstrap_ci(v.to_numpy(), seed=args.seed, **smd.BOOT)
            row.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi})
        row.update({"false_alarms_per_day": pooled.false_alarms_per_day.mean(),
                    "false_alarms_per_day_lo": float("nan"), "false_alarms_per_day_hi": float("nan"),
                    "runtime_per_window_ms_mean": 1000 * pooled.feature_s_per_window.mean(),
                    "runtime_per_window_ms_median": float("nan"),
                    "val_alarm_rate_mean": pooled.val_alarm_rate.mean()})
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)

    meta = run_metadata(seed=args.seed, config={"sources": [p for _, p in PARTS] + [str(own)]},
                        extra={"note": "E2 table; seeds are those of each source",
                               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    save_result(summary, Path("results/E2m/E2_table.csv"), meta)
    t = summary.copy()
    for m in ("pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day"):
        t[m] = [f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]" if r[m + "_lo"] == r[m + "_lo"] else f"{r[m]:.3f} (pooled)"
                for _, r in summary.iterrows()]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t[["feature_set", "windows", "pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day",
             "runtime_per_window_ms_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
