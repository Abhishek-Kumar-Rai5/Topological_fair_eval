"""E5b: k-channel injections and the baseline rule, on toy data (no real data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src import e5b
from src.injections import LENGTH, channel_profile
from src.injections_k import K, apply_plan_k, choose_units, n_eligible, plan_sites_k

N_CH = 12


def toy_series(n, rng):
    """6 coupled pairs: channels (2g, 2g+1) share a sine of period 18 + 3g, so every channel is periodic and coupled."""
    t = np.arange(n)
    cols = []
    for g in range(N_CH // 2):
        a = np.sin(2 * np.pi * t / (18 + 3 * g))
        cols += [a, a + 0.05 * rng.normal(size=n)]
    return 0.5 + 0.2 * np.column_stack(cols)


def make_smd(root, name="machine-1-1", n_ch=N_CH, n_train=2500, n_test=3000):
    for sub in ("train", "test", "test_label"):
        Path(root, sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train, test = toy_series(n_train, rng)[:, :n_ch], toy_series(n_test, rng)[:, :n_ch]
    labels = np.zeros(n_test, int)
    for a in (800, 2000):
        test[a:a + 40] += 0.4
        labels[a:a + 40] = 1
    np.savetxt(Path(root, "train", f"{name}.txt"), train, delimiter=",")
    np.savetxt(Path(root, "test", f"{name}.txt"), test, delimiter=",")
    np.savetxt(Path(root, "test_label", f"{name}.txt"), labels, fmt="%d")
    return train, test, labels


def profile(n_ch=N_CH):
    return channel_profile(toy_series(2500, np.random.default_rng(0))[:, :n_ch])


def test_k_distinct_channels_per_site_for_every_type():
    prof = profile()
    for itype in ("S1", "S2", "S3", "V1", "V2", "V3"):
        units = choose_units(itype, prof, K, np.random.default_rng(1))
        assert units is not None and len(units) == K
        if itype in ("S2", "S3"):                       # disjoint pairs: no channel is target and partner
            targets, partners = [u[0] for u in units], [u[1] for u in units]
            assert len(set(targets)) == K and not set(targets) & set(partners)
        else:
            assert len(set(units)) == K


def test_fewer_than_k_channels_is_skipped_and_reported():
    prof = profile(n_ch=4)
    plan = plan_sites_k(np.zeros(3000, int), prof, "V1", seed=3)
    assert plan.n_placed == 0 and "fewer than 5" in plan.reason
    assert n_eligible("V1", prof) == 4


def test_every_site_touches_exactly_k_channels_and_nothing_else():
    prof = profile()
    labels = np.zeros(3000, int)
    test = toy_series(3000, np.random.default_rng(5))
    for itype, mag in (("S1", 2), ("S2", 0.25), ("S3", 1.0), ("V1", 2), ("V2", 5), ("V3", 2)):
        plan = plan_sites_k(labels, prof, itype, seed=7, n=4)
        assert plan.n_placed == 4
        out, mask = apply_plan_k(test, plan, mag, prof)
        changed = np.abs(out - test) > 0
        assert not changed[~mask].any()                          # nothing outside the 60-sample sites
        for s in plan.starts:
            assert changed[s:s + LENGTH].any(axis=0).sum() <= K  # at most K channels per site (V2: spikes only)
            if itype in ("V1", "V3", "S1"):
                assert changed[s:s + LENGTH].any(axis=0).sum() == K
        assert len(mask) == len(test) and mask.sum() == 4 * LENGTH


def test_same_sites_for_every_magnitude_and_deterministic():
    prof, labels = profile(), np.zeros(3000, int)
    a = plan_sites_k(labels, prof, "V2", seed=11)
    b = plan_sites_k(labels, prof, "V2", seed=11)
    assert a.starts.tolist() == b.starts.tolist() and a.units == b.units


def config(root):
    cfg = yaml.safe_load(Path("configs/e5b_injections.yaml").read_text())
    cfg["data_root"] = str(root)
    cfg["seeds"] = [0]
    cfg["detector"]["n_estimators"] = 30
    cfg["injection"]["per_type_magnitude_machine"] = 3
    return cfg


def test_run_machine_end_to_end(tmp_path):
    make_smd(tmp_path / "SMD")
    df, place = e5b.run_machine(config(tmp_path / "SMD"), "machine-1-1")
    assert set(place.type) == {"S1", "S2", "S3", "V1", "V2", "V3"} and not place.skipped.any()
    assert (df.k == 5).all() and set(df.feature_set) == {"F1+F2", "F1+F2+F3"}
    assert (df.val_alarm_rate <= 0.01).all() and df.n_positive.min() > 0


def test_machine_with_too_few_channels_is_written_as_skipped(tmp_path):
    make_smd(tmp_path / "SMD", n_ch=4)
    df = e5b.run_machine_and_save(config(tmp_path / "SMD"), "machine-1-1", tmp_path / "out")
    place = pd.read_csv(tmp_path / "out" / "by_machine" / "machine-1-1_placement.csv")
    assert len(df) == 0 and place.skipped.all() and place.reason.str.contains("fewer than 5").all()
    assert (tmp_path / "out" / "by_machine" / "machine-1-1.csv").exists()


def test_baseline_rule_marks_failing_cells():
    from summarize_e5b import FAILS, apply_baseline_rule
    summary = pd.DataFrame({"type": ["S1", "S1", "V1", "V1"], "magnitude": [2, 2, 2, 2],
                            "feature_set": ["F1+F2", "F1+F2+F3", "F1+F2", "F1+F2+F3"],
                            "roc_auc": [0.55, 0.56, 0.82, 0.85], "pr_auc": [0.1, 0.1, 0.4, 0.5]})
    verdicts = pd.DataFrame({"type": ["S1", "V1"], "magnitude": [2, 2], "verdict": ["No added value", "Helps"]})
    summary["magnitude"] = summary["magnitude"].astype(float); verdicts["magnitude"] = verdicts["magnitude"].astype(float)
    v = apply_baseline_rule(summary, verdicts, 0.7)
    assert v.counts.tolist() == [False, True]
    assert v.verdict.tolist() == [FAILS, "Helps"] and v.verdict_if_counted.tolist() == ["No added value", "Helps"]
