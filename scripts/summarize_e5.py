"""Summarise E5 (results/E5/E5_results.csv) under prereg §6 and §8.

    python scripts/summarize_e5.py results/E5 [--seed 0]

Reuses scripts/summarize_smd.py (merged from master): the same per-cell table (PR-AUC, ROC-AUC,
recall, false alarms per day, each with a 95% cluster bootstrap CI over machines) and the same
Helps / No added value / Inconclusive rule, applied to every (type, magnitude) cell, with the
cell's own machines (machines with no eligible channel for a type are simply absent from it).

Writes next to the results (each with a .meta.json):
    E5_summary.csv      one row per type x magnitude x feature set  (the detection-vs-magnitude data)
    E5_verdicts.csv     Δ = PR-AUC(F1+F2+F3) - PR-AUC(F1+F2) per type x magnitude, with verdict;
                        decision-magnitude volume cells also get the Bonferroni check for P2
    E5_predictions.csv  P1 and P2 (prereg §8), from the decision-magnitude cells only
    E5_curves.png       PR-AUC against magnitude, one panel per type
Prints every machine x type where fewer than 10 injections fit (from E5_placement.csv).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import summarize_smd as smd                          # noqa: E402  (scripts/summarize_smd.py)
from src.bootstrap import average_over_seeds, paired_delta_ci  # noqa: E402
from src.config import load_config               # noqa: E402
from src.metadata import run_metadata, save_result  # noqa: E402

A, B = "F1+F2+F3", "F1+F2"
BONFERRONI_ALPHA = 0.05 / 3                       # prereg §8: P2, three volume types
MIN_VOLUME_DELTA = 0.02


def cells(df):
    for (t, m), g in df.groupby(["type", "magnitude"], sort=False):
        yield t, m, g.assign(runtime_per_window_ms=g["score_ms_per_window"])   # column summarize() expects


def summarize_cells(df, seed):
    summ, verd = [], []
    for t, m, g in cells(df):
        s, pos = smd.summarize(g, seed)
        s.insert(0, "magnitude", m); s.insert(0, "type", t); s.insert(1, "kind", g["kind"].iloc[0])
        summ.append(s)
        v = smd.verdict(g, A, B, pos, seed)
        decision = bool(g["is_decision"].iloc[0])
        v = {"type": t, "kind": g["kind"].iloc[0], "magnitude": m, "is_decision": decision, **v}
        if decision and v["kind"] == "volume":    # P2 check: CI at 1 - 0.05/3, and Δ >= 0.02
            pa = average_over_seeds(g[g.feature_set == A], "machine", "pr_auc").loc[pos]
            pb = average_over_seeds(g[g.feature_set == B], "machine", "pr_auc").loc[pos]
            d, lo, hi = paired_delta_ci(pa, pb, seed=seed, n_resamples=2000, ci=1 - BONFERRONI_ALPHA)
            v.update({"bonf_ci_lo": lo, "bonf_ci_hi": hi, "bonf_helps": bool(d >= MIN_VOLUME_DELTA and lo > 0)})
        verd.append(v)
    return pd.concat(summ, ignore_index=True), pd.DataFrame(verd)


def predictions(verdicts):
    """P1 and P2 from the decision-magnitude cells (prereg §8)."""
    d = verdicts[verdicts.is_decision].set_index("type")
    shape = d.loc[[t for t in ("S1", "S2", "S3") if t in d.index], "verdict"]
    vol = d.loc[[t for t in ("V1", "V2", "V3") if t in d.index]]
    helps, none = shape.str.startswith("Helps").sum(), (shape == "No added value").sum()
    p1 = ("Supported" if helps >= 2 else "Refuted" if none >= 2 else "Inconclusive")
    if len(shape) < 3:
        p1 += f" (only {len(shape)} of 3 shape types have a decision cell)"
    if vol.get("bonf_helps", pd.Series(dtype=bool)).fillna(False).any():
        p2 = "Refuted"
    elif len(vol) == 3 and (vol["verdict"] == "No added value").all():
        p2 = "Supported"
    else:
        p2 = "Inconclusive"
    return pd.DataFrame([
        {"prediction": "P1", "outcome": p1, "detail": f"shape types: " + ", ".join(f"{k}={v}" for k, v in shape.items())},
        {"prediction": "P2", "outcome": p2, "detail": f"volume types: " + ", ".join(f"{k}={v}" for k, v in vol["verdict"].items())}])


def plot_curves(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    types = [t for t in ("S1", "S2", "S3", "V1", "V2", "V3") if t in set(summary["type"])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, t in zip(axes.ravel(), types):
        for fs, g in summary[summary["type"] == t].groupby("feature_set"):
            g = g.sort_values("magnitude")
            ax.errorbar(g["magnitude"], g["pr_auc"], yerr=[g["pr_auc"] - g["pr_auc_lo"], g["pr_auc_hi"] - g["pr_auc"]],
                        marker="o", capsize=3, label=fs)
        ax.set_title(f"{t} ({summary[summary['type'] == t]['kind'].iloc[0]})"); ax.set_xlabel("magnitude")
    axes[0, 0].set_ylabel("PR-AUC"); axes[1, 0].set_ylabel("PR-AUC"); axes[0, 0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--seed", type=int, default=0, help="seed of the bootstrap resampling")
    args = ap.parse_args()
    folder = Path(args.folder)
    df = pd.read_csv(folder / "E5_results.csv")
    cfg = load_config("configs/e5_injections.yaml")
    summary, verdicts = summarize_cells(df, args.seed)
    preds = predictions(verdicts)
    meta = run_metadata(seed={"model_seeds": cfg["seeds"], "bootstrap_seed": args.seed}, config=cfg,
                        extra={"source": "E5_results.csv", "bootstrap": smd.BOOT})
    save_result(summary, folder / "E5_summary.csv", meta)
    save_result(verdicts, folder / "E5_verdicts.csv", meta)
    save_result(preds, folder / "E5_predictions.csv", meta)
    plot_curves(summary, folder / "E5_curves.png")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 120)
    print(f"\nE5: {df.machine.nunique()} machines\n")
    print(verdicts.round(4).to_string(index=False))
    print("\n", preds.to_string(index=False))
    place = pd.read_csv(folder / "E5_placement.csv")
    short = place[place.n_placed < place.n_requested]
    print(f"\nPLACEMENT: {len(short)} of {len(place)} machine x type cells have fewer than 10 injections")
    if len(short):
        print(short[["machine", "type", "n_placed", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
