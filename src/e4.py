"""E4 on CTU-13: leave-one-scenario-out, F1+F2 vs F1+F2+F3 and Z vs Z-control (prereg §2, §4, §7; notes 49-66).

Stage 1  build_scenario(n)      one scenario -> a table with a row per labelled host window:
                                the raw 60 x 6 window, and every feature that needs no fitting
                                (F1, F2's graph part, Z-control, Z).
Stage 2  run_fold(cfg, table, n) the scenario is the test set; per seed a validation scenario is drawn from
                                the others; F2's PCA and F3's reference distance are fitted on the training
                                windows (normal windows of the remaining scenarios) only; Isolation Forest,
                                threshold at 1% false alarms on the validation windows, metrics on the test windows.
Atomic writes and the validated resume come from src.metadata (same as E2/E3/E5).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import ctu13
from src import features_ctu13 as fc
from src.detector import IFDetector
from src.features_simple import F2Features, f1_features, f1_names, f2_names
from src.features_topo import F3Features, f3_names
from src.metadata import is_valid_result, run_metadata, save_result
from src.metrics import count_alarm_events, evaluate
from src.threshold import threshold_at_fpr

N_CH, SIZE = len(ctu13.CHANNELS), ctu13.WINDOW_BINS
F1_EXTRA = ["F1_unique_dst_ip", "F1_unique_dst_port", "F1_flow_count", "F1_bytes", "F1_packets"]
META = ["scenario", "host", "host_type", "window", "label", "n_edges_capped", "wtype"]
WTYPES = ["Shape", "Identity", "Mixed", "Unnamed"]   # per-window types from the flow-label strings (note 61)
SETS = ["F1+F2", "F1+F2+F3", "Z", "Z-control"]
F3_SPLIT = ["F1+F2+F3-H0", "F1+F2+F3-H1"]      # EXPLORATORY (E4x): F3's H0-only and H1-only halves; the same fitted F3
N_F3_HALF = 16                                   # F3 = 16 H0 summaries then 16 H1 summaries (src.features_topo)


def feature_columns():
    f1 = f1_names(N_CH) + F1_EXTRA
    return {"F1": f1, "G": fc.F2_GRAPH_NAMES, "ZC": fc.Z_CONTROL_NAMES, "Z": fc.z_names()}


# ------------------------------------------------------------------------------------------ stage 1
def host_rows(scenario, host, host_flows, is_infected, n_bins):
    """Rows (list of dicts) for the labelled windows of one host, with fit-free features."""
    js = ctu13.window_range(host_flows, n_bins)
    if len(js) == 0:
        return []
    wfs = [ctu13.window_flows(host_flows, j) for j in js]
    graphs, capped = zip(*[ctu13.window_graph(wf) for wf in wfs])
    dgm0, dgm1, n = fc.host_zigzag(list(graphs))                     # one zigzag per host
    z = fc.zigzag_features(dgm0, dgm1, n)
    rows = []
    for k, (j, wf) in enumerate(zip(js, wfs)):
        label = ctu13.window_label(wf, is_infected)
        if label < 0:
            continue                                                  # excluded window
        extra = [wf["dst"].nunique(), wf["dport"].nunique(), len(wf), wf["bytes"].sum(), wf["pkts"].sum()]
        rows.append({"scenario": scenario, "host": host, "host_type": "infected" if is_infected else "normal",
                     "window": int(j), "label": label, "n_edges_capped": bool(capped[k]),
                     "wtype": ctu13.window_type(wf) if label == 1 else "",
                     "_window": ctu13.channel_matrix(wf, j), "_extra": extra, "_graph": graphs[k], "_z": z[k]})
    return rows


def build_table(scenario, sc):
    """Scenario object (src.ctu13.Scenario) -> DataFrame with one row per labelled window."""
    rows = []
    for host in sc.infected + sc.normal:
        hf = sc.flows[sc.flows["host"] == host]
        rows += host_rows(scenario, host, hf, host in sc.infected, sc.n_bins)
    if not rows:
        return pd.DataFrame(columns=META)
    windows = np.stack([r["_window"] for r in rows])
    cols = feature_columns()
    f1 = np.hstack([f1_features(windows), np.array([r["_extra"] for r in rows], float)])
    gf = np.array([fc.f2_graph_features(r["_graph"]) for r in rows])
    zc = np.array([fc.z_control(r["_graph"]) for r in rows])
    zz = np.array([r["_z"] for r in rows])
    out = pd.DataFrame([{k: r[k] for k in META} for r in rows])
    out = pd.concat([out,
                     pd.DataFrame(windows.reshape(len(rows), -1).astype(np.float32), columns=[f"w{i}" for i in range(SIZE * N_CH)]),
                     pd.DataFrame(f1, columns=cols["F1"]), pd.DataFrame(gf, columns=cols["G"]),
                     pd.DataFrame(zc, columns=cols["ZC"]), pd.DataFrame(zz, columns=cols["Z"])], axis=1)
    return out


def build_scenario(cfg, scenario, out_dir):
    """Read one scenario, build its table, save by_scenario/<n>.parquet (atomic, with sidecar)."""
    t0 = time.time()
    sc = ctu13.read_scenario(ctu13.scenario_folder(scenario, cfg["data_root"]))
    table = build_table(scenario, sc)
    meta = run_metadata(seed=None, config=cfg, extra={
        "scenario": scenario, "seconds": round(time.time() - t0, 1), "n_bins": sc.n_bins,
        "infected_hosts": sc.infected, "normal_hosts": sc.normal, "t0": str(sc.t0),
        "n_windows_positive": int((table["label"] == 1).sum()), "n_windows_normal": int((table["label"] == 0).sum()),
        "n_windows_edge_capped": int(table["n_edges_capped"].sum())})
    assert len(table) > 0, f"scenario {scenario}: no labelled windows"
    save_result(table, Path(out_dir) / "by_scenario" / f"{scenario}.parquet", meta)
    return table


# ------------------------------------------------------------------------------------------ stage 2
def windows_of(df):
    return df[[f"w{i}" for i in range(SIZE * N_CH)]].to_numpy(float).reshape(len(df), SIZE, N_CH)


def draw_validation(seed, test_scenario, scenarios):
    others = sorted(s for s in scenarios if s != test_scenario)
    return int(np.random.default_rng([seed, test_scenario]).choice(others))


def false_alarm_events(df_test, alarm):
    """Alarms on normal windows, a run of consecutive windows of one host = 1 event; summed over hosts."""
    total = 0
    for _, g in df_test.assign(a=alarm & (df_test["label"].to_numpy() == 0)).sort_values(["host", "window"]).groupby("host"):
        total += count_alarm_events(g["a"].to_numpy())
    return total


def run_fold(cfg, tables, test_scenario, typed=False):
    """All result rows (every seed, every feature set) for one held-out scenario.

    typed=True also returns the per-type cells (note 61): for each window type T present among the scenario's
    positives, the metrics on its T-positives plus ALL its normal windows (other positives left out).
    """
    test = tables[test_scenario].reset_index(drop=True)
    cols = feature_columns()
    labels = test["label"].to_numpy()
    n_pos, n_norm = int((labels == 1).sum()), int((labels == 0).sum())
    assert n_pos > 0 and n_norm > 0, f"scenario {test_scenario}: needs positive and normal windows"
    scenarios = sorted(tables)
    days = (tables[test_scenario].attrs.get("n_bins", 0) or 0) * ctu13.BIN_SECONDS / 86400
    n_hosts = int(test.loc[test["host_type"] == "normal", "host"].nunique())
    fpr = cfg["threshold"]["false_alarm_rate"]
    det_cfg = cfg["detector"]
    rows, typed_rows = [], []
    wt = test["wtype"].to_numpy()
    for seed in cfg["seeds"]:
        val_s = draw_validation(seed, test_scenario, scenarios)
        train_s = [s for s in scenarios if s not in (test_scenario, val_s)]
        train = pd.concat([tables[s][tables[s]["label"] == 0] for s in train_s], ignore_index=True)
        val = tables[val_s][tables[val_s]["label"] == 0].reset_index(drop=True)
        assert len(train) > 0 and len(val) > 0, (test_scenario, seed)
        t_fit = time.perf_counter()
        w_tr = windows_of(train)
        f2, f3 = F2Features().fit(w_tr), F3Features().fit(w_tr)                 # training windows only

        def feats(df, w):
            parts = {"F1": df[cols["F1"]].to_numpy(float), "G": df[cols["G"]].to_numpy(float),
                     "F2": f2.transform(w)}
            return {"F1+F2": np.hstack([parts["F1"], parts["F2"], parts["G"]]),
                    "Z": df[cols["Z"]].to_numpy(float), "Z-control": df[cols["ZC"]].to_numpy(float),
                    "_F3": f3.transform(w)}
        ftr, fva = feats(train, w_tr), feats(val, windows_of(val))
        t_test = time.perf_counter()
        fte = feats(test, windows_of(test))
        per_window_feat = (time.perf_counter() - t_test) / len(test)
        for X in (ftr, fva, fte):
            f3x = X.pop("_F3")
            X["F1+F2+F3"] = np.hstack([X["F1+F2"], f3x])
            X["F1+F2+F3-H0"] = np.hstack([X["F1+F2"], f3x[:, :N_F3_HALF]])          # exploratory only
            X["F1+F2+F3-H1"] = np.hstack([X["F1+F2"], f3x[:, N_F3_HALF:]])
        for name in cfg.get("feature_sets", SETS):
            det = IFDetector(seed, det_cfg["n_estimators"], det_cfg.get("max_samples", "auto")).fit(ftr[name])
            val_scores = det.score(fva[name])
            thr = threshold_at_fpr(val_scores, fpr)                                # validation only
            t0 = time.perf_counter()
            scores = det.score(fte[name])
            secs = (time.perf_counter() - t0) / len(scores)
            r = evaluate(scores, labels, thr, 1, 1.0)
            events = false_alarm_events(test, scores > thr)
            r["false_alarm_events"] = events
            r["false_alarms_per_day"] = events / days / max(n_hosts, 1) if days > 0 else float("nan")
            row = ({"experiment": cfg["experiment"], "machine": f"scenario-{test_scenario}", "scenario": test_scenario,
                         "seed": seed, "feature_set": name, "n_features": ftr[name].shape[1], "threshold": thr,
                         "validation_scenario": val_s, "n_train_windows": len(train), "n_val_windows": len(val),
                         "n_test_windows": len(test), "val_alarm_rate": float((val_scores > thr).mean()),
                         "runtime_per_window_ms": 1000 * (secs + (per_window_feat if name.startswith("F1+F2") else 0)),
                         "fit_seconds": time.perf_counter() - t_fit, **r})
            rows.append(row)
            if typed:
                for T in WTYPES:
                    n_t = int((wt == T).sum())
                    if n_t == 0:
                        continue
                    keep = (labels == 0) | (wt == T)
                    rt = evaluate(scores[keep], labels[keep], thr, 1, 1.0)
                    typed_rows.append({**row, "wtype": T, "pr_auc": rt["pr_auc"], "roc_auc": rt["roc_auc"],
                                       "recall_at_thr": rt["recall_at_thr"], "n_positive": n_t, "n_normal": n_norm})
    return (pd.DataFrame(rows), pd.DataFrame(typed_rows)) if typed else pd.DataFrame(rows)


def load_tables(out_dir, scenarios):   # E4x passes cfg["tables_from"] = results/E4 here
    """Read the stage-1 tables (each must be valid) and attach the scenario length from its sidecar."""
    import json
    tables = {}
    for s in scenarios:
        p = Path(out_dir) / "by_scenario" / f"{s}.parquet"
        assert is_valid_result(p), f"stage-1 file missing or invalid: {p}"
        t = pd.read_parquet(p)
        t.attrs["n_bins"] = json.loads(p.with_name(f"{s}.meta.json").read_text())["n_bins"]
        tables[s] = t
    return tables


def run_fold_and_save(cfg, tables, test_scenario, out_dir):
    df, typed = run_fold(cfg, tables, test_scenario, typed=True)
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"scenario": test_scenario})
    save_result(typed, Path(out_dir) / "by_fold" / f"{test_scenario}_wtype.csv", meta)
    save_result(df, Path(out_dir) / "by_fold" / f"{test_scenario}.csv", meta)        # written last: marks the fold done
    return df


def merge_results(cfg, scenarios, out_dir):
    out_dir = Path(out_dir)
    df = pd.concat([pd.read_csv(out_dir / "by_fold" / f"{s}.csv") for s in scenarios], ignore_index=True)
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"scenarios": list(scenarios)})
    typed = [pd.read_csv(out_dir / "by_fold" / f"{s}_wtype.csv") for s in scenarios if (out_dir / "by_fold" / f"{s}_wtype.csv").exists()]
    if typed:
        save_result(pd.concat(typed, ignore_index=True), out_dir / f"{cfg['experiment']}_wtype_results.csv", meta)
    return save_result(df, out_dir / f"{cfg['experiment']}_results.csv", meta)


def usable_scenarios(cfg):
    """The 12 scenarios of the data folder (13 minus scenario 10, which has no .binetflow)."""
    root = Path(cfg["data_root"])
    out = []
    for s in range(1, 14):
        f = ctu13.scenario_folder(s, root)
        if f.exists() and list(f.glob("*.binetflow*")):
            out.append(s)
    return out
