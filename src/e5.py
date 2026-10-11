"""E5 runner: shape vs volume injections into SMD normal test segments (prereg §3, §7).

Per machine:
    1. windows and features as in E2/E3: groups fitted on TRAINING windows only; one detector per
       (seed, feature set), fitted on training features; threshold = 1% false alarms on VALIDATION.
       Injections never touch training or validation, so the detector and threshold are the same
       for every injected series.
    2. eligible channels come from the training part (first 80%) via channel_profile.
    3. for each type and magnitude: inject into a copy of the test series, relabel the windows
       (src.injections.injection_window_labels), score, evaluate.
Speed: a window's features depend only on that window, so only windows that overlap an injection
are recomputed; every other window reuses the features of the unmodified test series (exact).
Output rows: one per machine x type x magnitude x seed x feature set. Placement (how many of the 10
injections fit, and why not) is written separately, so no skipped cell is silent.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.detector import IFDetector
from src.feature_sets import GROUPS
from src.injections import (KIND, LENGTH, TYPES, apply_plan, channel_profile, injection_window_labels,
                            machine_seed, plan_sites)
from src.metadata import run_metadata, save_result
from src.metrics import evaluate
from src.prepare import prepare_machine
from src.smd import load_machine
from src.splits import split_train_val
from src.threshold import threshold_at_fpr
from src.windows import make_windows


def feature_function(groups, mw):
    """Fit the groups on training windows; return (features of windows, dict of train/val features)."""
    fitted = {g: GROUPS[g][0]().fit(mw.train) for g in groups}

    def feats(windows):
        return {g: fitted[g].transform(windows) for g in groups}
    return feats


def stack(feats, names):
    return np.hstack([feats[p] for p in names.split("+")])


def injected_test_features(test_series, base_windows, base_feats, starts, mask, feats_fn, size=60):
    """Features of the windows of the injected series, recomputing only windows that overlap `mask`."""
    touched = np.array([mask[s:s + size].any() for s in starts])
    out = {g: f.copy() for g, f in base_feats.items()}
    if touched.any():
        idx = np.flatnonzero(touched)
        win = test_series[starts[idx][:, None] + np.arange(size)[None, :]]
        new = feats_fn(win)
        for g in out:
            out[g][idx] = new[g]
    return out


def run_machine(cfg, machine):
    """(results DataFrame, placement DataFrame) for one machine."""
    inj_cfg, w = cfg["injection"], cfg["windows"]
    det_cfg = cfg["detector"]
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
        plan = plan_sites(real_point_labels, prof, itype, machine_seed(machine, itype),
                          n=n_req, min_gap=inj_cfg["min_spacing"])
        placement.append({"machine": machine, "type": itype, "kind": KIND[itype],
                          "n_requested": n_req, "n_placed": plan.n_placed, "reason": plan.reason,
                          "n_periodic": len(prof.periodic), "n_pairs": len(prof.pairs),
                          "n_volume_channels": len(prof.volume)})
        if plan.n_placed == 0:
            continue
        for mag in spec["magnitudes"]:
            series, mask = apply_plan(test_raw, plan, mag, prof)
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
                                 "seed": seed, "feature_set": n, "threshold": thr[n, seed],
                                 "val_alarm_rate": val_rate[n, seed], "n_sites": plan.n_placed,
                                 "score_ms_per_window": 1000 * secs, **r})
    return pd.DataFrame(rows), pd.DataFrame(placement)


def run_machine_and_save(cfg, machine, out_dir):
    """run_machine, then save by_machine/<machine>.csv and <machine>_placement.csv (atomic, with sidecars)."""
    df, placement = run_machine(cfg, machine)
    assert len(df) > 0, f"{machine}: no injection could be placed for any type"
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"machine": machine})
    out = Path(out_dir) / "by_machine"
    save_result(placement, out / f"{machine}_placement.csv", meta)
    save_result(df, out / f"{machine}.csv", meta)          # written last: it marks the machine as done
    return df


def merge_results(cfg, machines, out_dir):
    """Join per-machine files into E5_results.csv and E5_placement.csv (+ metadata)."""
    out_dir = Path(out_dir)
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"machines": list(machines)})
    place = pd.concat([pd.read_csv(out_dir / "by_machine" / f"{m}_placement.csv") for m in machines], ignore_index=True)
    save_result(place, out_dir / f"{cfg['experiment']}_placement.csv", meta)
    df = pd.concat([pd.read_csv(out_dir / "by_machine" / f"{m}.csv") for m in machines], ignore_index=True)
    return save_result(df, out_dir / f"{cfg['experiment']}_results.csv", meta)
