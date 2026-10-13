"""Does Topology Help? Streamlit demo.

Run:  streamlit run app.py

Reads only files that already exist (results/, literature/, docs/), plus ONE live computation (the Live page, which
reads the SMD data in data/SMD, or the folder in the SMD_ROOT environment variable). Nothing else is recomputed.
Every page says where its numbers come from.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import app_helpers as H

ROOT = H.ROOT
PAGES = ["Audit", "Code audit", "E2 and E3", "E3x: H0 against H1", "E4: zigzag against its control",
         "E5 and E5b curves", "Live: one window"]
BLUE, RED, GREY = "#1f77b4", "#d62728", "#7f7f7f"

st.set_page_config(page_title="Does Topology Help?", layout="wide")


# ------------------------------------------------------------------------------------------------ small helpers
@st.cache_data
def load(rel_path):
    return H.read_csv(rel_path)


def need(df, rel_path):
    """True if the file was found; otherwise tell the user which file is missing."""
    if df is None:
        st.info(f"`{rel_path}` is not in this checkout, so this part is empty.")
        return False
    return True


def ci_text(df, metric):
    """'0.461 [0.390, 0.540]' strings for a summary table (mean and 95% CI over machines)."""
    return [f"{r[metric]:.3f} [{r[metric + '_lo']:.3f}, {r[metric + '_hi']:.3f}]" for _, r in df.iterrows()]


def summary_table(df):
    t = pd.DataFrame({"feature set": df["feature_set"]})
    for m, label in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("recall_at_thr", "recall at 1% FA"),
                     ("false_alarms_per_day", "false alarms / day")):
        t[label] = ci_text(df, m)
    t["runtime ms/window"] = df["runtime_per_window_ms_mean"].round(3)
    return t


def verdict_table(df):
    cols = [c for c in ("comparison", "delta_pr_auc", "ci_lo", "ci_hi", "verdict",
                        "pooled_positive_windows", "pooled_normal_windows") if c in df.columns]
    return df[cols].round(4)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def delta_bar(delta, title, ylabel="Δ PR-AUC"):
    """One bar per machine (or scenario), sorted; blue = topology better, red = worse."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(range(len(delta)), delta.values, color=[BLUE if v > 0 else RED for v in delta.values])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(delta)))
    ax.set_xticklabels([str(i).replace("machine-", "").replace("scenario-", "S") for i in delta.index], rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ pages
def page_audit():
    st.header("Audit: what the published evidence looks like")
    st.caption("Source: `literature/tda_records_master.csv` (95 records; 32 core papers, D03 excluded) and `literature/figures/`.")
    cols = st.columns(2)
    for col, name, cap in ((cols[0], "fig1_evidence_map.png", "Figure 1: outcomes by cluster and by whether the comparison isolates topology"),
                           (cols[1], "fig2_rigor_heatmap.png", "Figure 2: the 13 rigour checks for each core paper")):
        p = ROOT / "literature" / "figures" / name
        if p.exists():
            col.image(str(p), caption=cap, width="stretch")
        else:
            col.info(f"`literature/figures/{name}` is missing.")
    master = load("literature/tda_records_master.csv")
    if not need(master, "literature/tda_records_master.csv"):
        return
    core = H.core_papers(master)
    st.subheader(f"The {len(core)} core papers")
    c1, c2 = st.columns(2)
    outcome = c1.multiselect("Outcome (OUT)", sorted(core["OUT"].dropna().unique()))
    cluster = c2.multiselect("Cluster", sorted(core["cluster"].dropna().unique()))
    view = core
    if outcome:
        view = view[view["OUT"].isin(outcome)]
    if cluster:
        view = view[view["cluster"].isin(cluster)]
    st.dataframe(view, width="stretch", hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("link"), "assessment": st.column_config.TextColumn(width="large")})
    st.caption(f"{len(view)} of {len(core)} papers shown. OUT = outcome of the comparison, ISO = whether it isolates topology, "
               "USE = what topology is used for, verification = how much of the paper was read.")


def page_code_audit():
    st.header("Code-audit findings")
    st.caption("Source: `docs/code_verification.md`. Each finding is tied to a repository commit and line numbers.")
    text = H.code_audit_text()
    if text is None:
        st.info("`docs/code_verification.md` is missing.")
    else:
        st.markdown(text)


def page_e2_e3():
    st.header("E2 and E3: does topology help on SMD?")
    st.caption("Sources: `results/E3/` (our pipelines, 28 machines × 5 seeds), `results/E2m/E2_table.csv` (TADA, PHANTOM, TAME). "
               "Threshold = 1% false alarms on validation windows; no point adjustment; CIs are 95% bootstrap over machines.")
    st.subheader("E3: same detector, different features")
    summ, verd = load("results/E3/E3_summary.csv"), load("results/E3/E3_verdicts.csv")
    if need(summ, "results/E3/E3_summary.csv"):
        st.dataframe(summary_table(summ), hide_index=True, width="stretch")
    if need(verd, "results/E3/E3_verdicts.csv"):
        st.dataframe(verdict_table(verd), hide_index=True, width="stretch")
    st.subheader("E2: our pipelines next to the published methods")
    e2 = load("results/E2m/E2_table.csv")
    if need(e2, "results/E2m/E2_table.csv"):
        t = summary_table(e2)
        t.insert(1, "windows", e2["windows"])
        st.dataframe(t, hide_index=True, width="stretch")
        st.caption("TAME on its own windows uses 200-sample windows, so its false alarms per day are pooled over one stream.")
    st.subheader("Per-machine Δ")
    res = load("results/E3/E3_results.csv")
    if need(res, "results/E3/E3_results.csv"):
        choice = st.radio("Comparison", ["Main gain Δ = F1+F2+F3 − F1+F2", "Swap gain Δs = F1+F3 − F1+F2"], horizontal=True)
        a, b = ("F1+F2+F3", "F1+F2") if choice.startswith("Main") else ("F1+F3", "F1+F2")
        delta = H.per_machine_delta(res, a, b)
        show(delta_bar(delta, f"{a} − {b}: PR-AUC difference per machine (mean over 5 seeds)"))
        st.write(f"Δ > 0 on **{int((delta > 0).sum())} of {len(delta)}** machines (mean Δ = {delta.mean():+.3f}).")


def page_e3x():
    st.header("E3x (exploratory): which homology dimension carries the gain?")
    st.caption("Source: `results/E3x/`. F3 split into its H0 half (connected components) and H1 half (loops), each added to F1+F2. "
               "Exploratory, not in the preregistration; no prediction is decided from it.")
    summ, verd = load("results/E3x/E3x_summary.csv"), load("results/E3x/E3x_verdicts.csv")
    res = load("results/E3x/E3x_results.csv")
    if not (need(summ, "results/E3x/E3x_summary.csv") and need(verd, "results/E3x/E3x_verdicts.csv")):
        return
    st.dataframe(summary_table(summ), hide_index=True, width="stretch")
    st.dataframe(verdict_table(verd), hide_index=True, width="stretch")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(summ))
    ax.errorbar(summ["pr_auc"], y, xerr=[summ["pr_auc"] - summ["pr_auc_lo"], summ["pr_auc_hi"] - summ["pr_auc"]],
                fmt="o", color=BLUE, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(summ["feature_set"])
    ax.set_xlabel("PR-AUC (mean over machines, 95% CI)")
    fig.tight_layout()
    show(fig)
    if need(res, "results/E3x/E3x_results.csv"):
        d0 = H.per_machine_delta(res, "F1+F2+F3-H0", "F1+F2")
        d1 = H.per_machine_delta(res, "F1+F2+F3-H1", "F1+F2").reindex(d0.index)
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.scatter(d0, d1, color=BLUE, s=18)
        lim = [min(d0.min(), d1.min()) - 0.01, max(d0.max(), d1.max()) + 0.01]
        ax.plot(lim, lim, color=GREY, lw=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Δ PR-AUC with H0 only")
        ax.set_ylabel("Δ PR-AUC with H1 only")
        ax.set_title("one point per machine (above the line: H1 gains more)", fontsize=9)
        fig.tight_layout()
        show(fig)
        st.write(f"H0-only beats H1-only on **{int((d0 > d1).sum())} of {len(d0)}** machines.")


def page_e4():
    st.header("E4: zigzag against its control (CTU-13)")
    st.caption("Source: `results/E4/`. Leave-one-scenario-out over 12 scenarios, per-host windows. Z = zigzag persistence features; "
               "Z-control = the simple graph counts (edges, vertices, components, diameter). CIs are bootstrap over scenarios.")
    summ, verd = load("results/E4/E4_summary.csv"), load("results/E4/E4_verdicts.csv")
    res = load("results/E4/E4_results.csv")
    if not (need(summ, "results/E4/E4_summary.csv") and need(verd, "results/E4/E4_verdicts.csv")):
        return
    st.dataframe(summary_table(summ), hide_index=True, width="stretch")
    st.dataframe(verdict_table(verd), hide_index=True, width="stretch")
    st.caption("The prereg verdict words have no case for 'clearly worse', so a CI entirely below 0 is still printed as Inconclusive.")
    if need(res, "results/E4/E4_results.csv"):
        mean = res.groupby(["scenario", "feature_set"])["pr_auc"].mean().unstack()
        fig, ax = plt.subplots(figsize=(10, 3.4))
        x = np.arange(len(mean))
        ax.bar(x - 0.2, mean["Z-control"], 0.4, label="Z-control", color=GREY)
        ax.bar(x + 0.2, mean["Z"], 0.4, label="Z (zigzag)", color=BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in mean.index])
        ax.set_ylabel("PR-AUC (mean over 5 seeds)")
        ax.legend()
        fig.tight_layout()
        show(fig)
        better = int((mean["Z"] > mean["Z-control"]).sum())
        st.write(f"Z beats its control in **{better} of {len(mean)}** scenarios.")
    st.subheader("Cells by behaviour type")
    st.write("Primary typing: each positive window typed by the behaviours its botnet flows name (`CC<n>`, `P2P` = Shape, `SPAM` = Identity). "
             "Sensitivity typing: the scenario's README / Table 2. **Neither can test P2** (the strings never name port scan, DDoS or "
             "click fraud, and no scenario is Volume-only), and neither decides a prediction.")
    for rel, title in (("results/E4/E4_by_wtype.csv", "Primary: by window type (flow-label strings)"),
                       ("results/E4/E4_by_type.csv", "Sensitivity: by scenario type (README / Table 2)")):
        df = load(rel)
        st.markdown(f"**{title}**")
        if need(df, rel):
            keep = [c for c in ("wtype", "type", "n_scenarios", "comparison", "delta_pr_auc", "ci_lo", "ci_hi", "verdict",
                                "pooled_positive_windows", "pooled_normal_windows") if c in df.columns]
            st.dataframe(df[keep].round(4), hide_index=True, width="stretch")


def injection_curves(summary, base_rate, title):
    types = [t for t in ("S1", "S2", "S3", "V1", "V2", "V3") if t in set(summary["type"])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey=True)
    for ax, t in zip(axes.ravel(), types):
        for fs, color in (("F1+F2", GREY), ("F1+F2+F3", BLUE)):
            g = summary[(summary["type"] == t) & (summary["feature_set"] == fs)].sort_values("magnitude")
            ax.errorbar(g["magnitude"], g["pr_auc"], yerr=[g["pr_auc"] - g["pr_auc_lo"], g["pr_auc_hi"] - g["pr_auc"]],
                        marker="o", capsize=3, color=color, label=fs)
        ax.axhline(base_rate, color=RED, ls="--", lw=0.9, label="chance (base rate)")
        kind = summary.loc[summary["type"] == t, "kind"].iloc[0]
        ax.set_title(f"{t} ({kind})", fontsize=10)
        ax.set_xlabel("magnitude")
    axes[0, 0].set_ylabel("PR-AUC")
    axes[1, 0].set_ylabel("PR-AUC")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def page_e5():
    st.header("E5 and E5b: injected anomalies, shape against volume")
    st.caption("Sources: `results/E5/`, `results/E5b/`. Synthetic anomalies are injected into normal test segments of each SMD machine. "
               "Curves show PR-AUC against injection magnitude; the dashed line is chance (the share of positive windows).")
    tab5, tab5b = st.tabs(["E5 (preregistered, 1 channel)", "E5b (exploratory, 5 channels)"])
    for tab, exp, extra in ((tab5, "E5", ""), (tab5b, "E5b", "b")):
        with tab:
            summ, res = load(f"results/{exp}/{exp}_summary.csv"), load(f"results/{exp}/{exp}_results.csv")
            verd = load(f"results/{exp}/{exp}_verdicts.csv")
            if not (need(summ, f"results/{exp}/{exp}_summary.csv") and need(res, f"results/{exp}/{exp}_results.csv")):
                continue
            base = H.pooled_base_rate(res)
            show(injection_curves(summ, base, f"{exp}: PR-AUC against magnitude (chance = {base:.3f})"))
            roc = summ[summ.feature_set == "F1+F2"]["roc_auc"]
            if exp == "E5":
                st.warning(f"The F1+F2 baseline is at chance: its ROC-AUC is between {roc.min():.2f} and {roc.max():.2f} in every cell, "
                           "so the preregistered E5 failed to detect the injections and its P1 / P2 verdicts are uninformative.")
            else:
                n_ok = int(verd["counts"].sum()) if verd is not None and "counts" in verd else 0
                st.warning(f"A cell counts only if the F1+F2 baseline ROC-AUC is at least 0.7: **{n_ok} of {len(verd) if verd is not None else 0}** "
                           "cells count. Cells that fail are listed with their numbers but get no verdict.")
            if need(verd, f"results/{exp}/{exp}_verdicts.csv"):
                keep = [c for c in ("type", "magnitude", "is_decision", "baseline_roc_auc", "counts", "delta_pr_auc", "ci_lo", "ci_hi",
                                    "verdict") if c in verd.columns]
                st.dataframe(verd[keep].round(4), hide_index=True, width="stretch")


def page_live():
    st.header("Live: the topology of one window")
    st.caption("The one live computation. It reads the SMD data (`data/SMD`, or the folder in `SMD_ROOT`), fits everything on the "
               "machine's training windows only (seed 0, threshold = 1% false alarms on validation windows) and shows what F3 sees "
               "in one test window. The label is displayed after scoring and is never used.")
    from src import live
    from src.prepare import prepare_machine
    from src.smd import list_machines

    root = Path(os.environ.get("SMD_ROOT", ROOT / "data" / "SMD"))
    if not (root / "train").exists():
        st.info(f"SMD data not found in `{root}`; this page needs it.")
        return
    machine = st.selectbox("Machine", list_machines(root))

    @st.cache_resource(show_spinner="Fitting features and detectors on the training windows ...")
    def prepared(name, root_str):
        mw = prepare_machine(name, root=root_str)
        return mw, live.fit_scorers(mw)

    mw, scorers = prepared(machine, str(root))
    kind = st.radio("Windows to browse", ["all", "positive (anomalous)", "normal"], horizontal=True)
    labels = mw.test_labels
    idx_all = np.arange(len(labels)) if kind == "all" else np.flatnonzero(labels == (1 if kind.startswith("positive") else 0))
    if len(idx_all) == 0:
        st.info("This machine has no windows of that kind.")
        return
    k = st.slider("Window", 0, len(idx_all) - 1, 0) if len(idx_all) > 1 else 0
    idx = int(idx_all[k])
    window = mw.test[idx]
    st.write(f"Test window **{idx}** (samples {mw.test_starts[idx]}–{mw.test_starts[idx] + window.shape[0] - 1}), "
             f"{window.shape[1]} channels.")

    view = live.window_view(scorers["groups"]["F3"], window)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Delay-embedded cloud (3-D PCA)")
        fig = plt.figure(figsize=(5, 4.4))
        ax = fig.add_subplot(projection="3d")
        p = view["cloud3"]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=GREY, lw=0.5)
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=np.arange(len(p)), cmap="viridis", s=14)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="time within the window")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        fig.tight_layout()
        show(fig)
        st.caption(f"{len(p)} points in {view['points'].shape[1]} dimensions, projected to 3; the 3 components keep "
                   f"{100 * view['explained'].sum():.0f}% of the variance. Persistence is computed on the full cloud, not on this projection.")
    with c2:
        st.subheader("Persistence diagrams")
        fig, axes = plt.subplots(1, 2, figsize=(6, 3.4))
        for ax, name, dgm in ((axes[0], "H0 (components)", view["dgm0"]), (axes[1], "H1 (loops)", view["dgm1"])):
            d = dgm[np.isfinite(dgm[:, 1])]
            hi = max(float(d.max()) if len(d) else 1.0, 1e-6) * 1.05
            ax.plot([0, hi], [0, hi], color=GREY, lw=0.7)
            ax.scatter(d[:, 0], d[:, 1], s=14, color=BLUE)
            ax.set_title(f"{name}: {len(d)} bars", fontsize=9)
            ax.set_xlabel("birth")
            ax.set_ylabel("death")
        fig.tight_layout()
        show(fig)
    st.subheader("Minimum spanning tree against H0")
    edges, deaths = view["mst_edges"], view["h0_deaths"]
    d = view["max_abs_diff"]
    tol = 1e-5 * max(float(deaths.max()) if len(deaths) else 1.0, 1.0)
    if d is not None and d <= tol:
        st.success(f"Identical to float32 precision: the {len(edges)} MST edge lengths equal the {len(deaths)} finite H0 death times "
                   f"(largest difference {d:.1e}). H0 persistence is exactly the single-linkage tree of the cloud.")
    else:
        st.error(f"Mismatch: {len(edges)} MST edges against {len(deaths)} H0 deaths (largest difference {d}).")
    if len(edges) == len(deaths):
        st.dataframe(pd.DataFrame({"rank": np.arange(1, len(edges) + 1), "MST edge length": edges,
                                   "H0 death": deaths, "difference": edges - deaths}), hide_index=True, height=200)
    st.subheader("Scores")
    scores = live.score_window(scorers, window)
    cols = st.columns(len(scores))
    for col, (name, (s, thr, alarm)) in zip(cols, scores.items()):
        col.metric(name, f"{s:.3f}", f"{s - thr:+.3f} vs threshold {thr:.3f}", delta_color="inverse")
        col.write("**ALARM**" if alarm else "no alarm")
    st.caption(f"Higher score = more anomalous; a window alarms when its score is above the threshold. "
               f"This window's label: {({1: 'positive (anomalous)', 0: 'normal', -1: 'excluded from the metrics'})[int(labels[idx])]}.")


# ------------------------------------------------------------------------------------------------ main
st.sidebar.title("Does Topology Help?")
st.sidebar.caption("An audit and fair re-evaluation of TDA anomaly detection for systems and security telemetry.")
page = st.sidebar.radio("Page", PAGES)
{"Audit": page_audit, "Code audit": page_code_audit, "E2 and E3": page_e2_e3, "E3x: H0 against H1": page_e3x,
 "E4: zigzag against its control": page_e4, "E5 and E5b curves": page_e5, "Live: one window": page_live}[page]()
