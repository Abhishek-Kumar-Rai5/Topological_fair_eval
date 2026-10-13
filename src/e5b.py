"""E5b runner (exploratory): the E5 pipeline with K = 5 channels per injection (src/injections_k.py).

Identical to src/e5.py except for the injection step: same windows, features (groups fitted on training
windows only), detector (Isolation Forest, fixed defaults), threshold (1% false alarms on validation windows),
labels and metrics. Output rows: one per machine x type x magnitude x seed x feature set. Placement says, for
every machine x type, how many sites fit and why not (skipped types are never silent).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.detector import IFDetector
from src.e5 import feature_function, injected_test_features, stack
from src.injections import KIND, TYPES, channel_profile, injection_window_labels, machine_seed
from src.injections_k import K, apply_plan_k, n_eligible, plan_sites_k
from src.metadata import run_metadata, save_result
from src.metrics import evaluate
from src.prepare import prepare_machine
from src.smd import load_machine
from src.splits import split_train_val
from src.threshold import threshold_at_fpr


def run_machine(cfg, machine):
    """(results DataFrame, placement DataFrame) for one machine."""
    inj_cfg, w, det_cfg = cfg["injection"], cfg["windows"], cfg["detector"]
    k = inj_cfg.get("k", K)
    assert det_cfg["name"] == "isolation_forest", det_cfg
    sets = [cfg["comparisons"]["main_gain"]["a"], cfg["comparisons"]["main_gain"]["b"]]
    groups = sorted({g for n in sets for g in n.split("+")})
    mw = prepare_machine(machine, root=cfg["data_root"], size=w["size"], step=w["step"],
                         tail=w["label_tail"], val_frac=cfg["split"]["val_frac"])
    train_raw, test_raw, real_point_labels = load_machine(machine, cfg["data_root"])
    fit_part, _ = split_train_val(train_raw, cfg["split"]["val_frac"])
    prof = channel_profile(fit_part, inj_cfg["eligibility"]["periodic"]["acf_at_dominant_lag_gt"],
                           inj_cfg["eligibility"]["coupled_pair"]["abs_corr_gt"])
    feats_fn = feature_function(groups, mw)
    f_train, f_val, f_base = feats_fn(mw.train), feats_fn(mw.val), feats_fn(mw.test)
    X = {n: {"train": stack(f_train, n), "val": stack(f_val, n)} for n in sets}
    dets, thr, val_rate = {}, {}, {}
    fpr = cfg["threshold"]["false_alarm_rate"]
    for n in sets:
        for seed in cfg["seeds"]:
            d = IFDetector(seed, det_cfg["n_estimators"], det_cfg.get("max_samples", "auto"))
            d.fit(X[n]["train"])
            vs = d.score(X[n]["val"])
            dets[n, seed], thr[n, seed] = d, threshold_at_fpr(vs, fpr)      # validation only
            val_rate[n, seed] = float((vs > thr[n, seed]).mean())

    rows, placement = [], []
    n_req = inj_cfg["per_type_magnitude_machine"]
    for itype in TYPES:
        spec = inj_cfg["types"][itype]
        plan = plan_sites_k(real_point_labels, prof, itype, machine_seed(machine, itype), k=k,
                            n=n_req, min_gap=inj_cfg["min_spacing"])
        placement.append({"machine": machine, "type": itype, "kind": KIND[itype], "k": k,
                          "n_requested": n_req, "n_placed": plan.n_placed, "reason": plan.reason,
                          "n_eligible_channels": n_eligible(itype, prof),
                          "skipped": plan.n_placed == 0})
        if plan.n_placed == 0:
            continue
        for mag in spec["magnitudes"]:
            series, mask = apply_plan_k(test_raw, plan, mag, prof)
            labels = injection_window_labels(real_point_labels, mask, mw.test_starts, w["size"], w["label_tail"])
            f_inj = injected_test_features(series, mw.test, f_base, mw.test_starts, mask, feats_fn, w["size"])
            for n in sets:
                Xt = stack(f_inj, n)
                for seed in cfg["seeds"]:
                    t0 = time.perf_counter()
                    scores = dets[n, seed].score(Xt)
                    secs = (time.perf_counter() - t0) / len(scores)
                    r = evaluate(scores, labels, thr[n, seed], mw.n_test_points, cfg.get("minutes_per_point", 1.0))
                    rows.append({"experiment": cfg["experiment"], "machine": machine, "type": itype,
                                 "kind": KIND[itype], "magnitude": mag, "is_decision": mag == spec["decision"],
                                 "k": k, "seed": seed, "feature_set": n, "threshold": thr[n, seed],
                                 "val_alarm_rate": val_rate[n, seed], "n_sites": plan.n_placed,
                                 "score_ms_per_window": 1000 * secs, **r})
    return pd.DataFrame(rows), pd.DataFrame(placement)


def run_machine_and_save(cfg, machine, out_dir):
    """run_machine, then save by_machine/<machine>.csv and <machine>_placement.csv (atomic, with sidecars)."""
    df, placement = run_machine(cfg, machine)
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"machine": machine})
    out = Path(out_dir) / "by_machine"
    save_result(placement, out / f"{machine}_placement.csv", meta)
    if len(df) == 0:                                  # every type skipped on this machine: still mark it done
        df = pd.DataFrame(columns=["experiment", "machine", "type", "kind", "magnitude", "is_decision", "k", "seed",
                                   "feature_set", "n_positive", "n_normal", "pr_auc", "roc_auc"])
    save_result(df, out / f"{machine}.csv", meta)     # written last: it marks the machine as done
    return df


def merge_results(cfg, machines, out_dir):
    """Join per-machine files into E5b_results.csv and E5b_placement.csv (+ metadata)."""
    out_dir = Path(out_dir)
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"machines": list(machines)})
    place = pd.concat([pd.read_csv(out_dir / "by_machine" / f"{m}_placement.csv") for m in machines], ignore_index=True)
    save_result(place, out_dir / f"{cfg['experiment']}_placement.csv", meta)
    frames = [pd.read_csv(out_dir / "by_machine" / f"{m}.csv") for m in machines]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    return save_result(df, out_dir / f"{cfg['experiment']}_results.csv", meta)
