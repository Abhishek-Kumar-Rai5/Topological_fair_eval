"""E5 injections (prereg §3) on toy data only."""
from pathlib import Path

import numpy as np
import yaml

from src.injections import (KIND, LENGTH, TYPES, apply_plan, channel_profile, dominant_lag,
                            injection_window_labels, machine_seed, plan_sites)


def toy(n_train=3000, n_test=4000, seed=0):
    """ch0 sine (period 20), ch1 = ch0 + noise (coupled), ch2 noise, ch3 constant, ch4 random walk."""
    rng = np.random.default_rng(seed)

    def make(n):
        t = np.arange(n)
        a = np.sin(2 * np.pi * t / 20) + 0.1 * rng.normal(size=n)
        return np.column_stack([a, a + 0.1 * rng.normal(size=n), rng.normal(size=n),
                                np.full(n, 0.5), np.cumsum(rng.normal(size=n))])
    return make(n_train), make(n_test)


def setup(itype, seed=1, labels=None):
    train, test = toy()
    prof = channel_profile(train)
    labels = np.zeros(len(test), int) if labels is None else labels
    return train, test, labels, prof, plan_sites(labels, prof, itype, seed)


def test_eligibility_from_training():
    train, _ = toy()
    prof = channel_profile(train)
    assert prof.periodic[0] == 20 and prof.periodic[1] == 20
    assert 2 not in prof.periodic and 3 not in prof.periodic and 4 not in prof.periodic  # noise, constant, trend
    assert (0, 1) in prof.pairs and 3 not in prof.nonconstant
    assert dominant_lag(np.arange(500.0)) is None


def test_config_lists_the_same_types_and_kinds():
    cfg = yaml.safe_load(Path("configs/e5_injections.yaml").read_text())["injection"]["types"]
    assert list(cfg) == TYPES
    assert {k: v["kind"] for k, v in cfg.items()} == KIND


def test_sites_ten_spaced_and_in_normal_segments():
    labels = np.zeros(4000, int)
    labels[1000:1200] = 1
    for t in TYPES:
        _, _, _, prof, plan = setup(t, labels=labels)
        assert plan.n_placed == 10 and plan.reason == "", t
        s = np.sort(plan.starts)
        assert (np.diff(s) >= LENGTH + 120).all(), t           # gap >= 120 between footprints
        assert all(labels[x:x + LENGTH].sum() == 0 for x in s), t   # never touch labelled points


def test_apply_changes_only_masked_points_and_input_is_unchanged():
    for t in TYPES:
        _, test, labels, prof, plan = setup(t)
        before = test.copy()
        out, mask = apply_plan(test, plan, {"S2": 0.25, "S3": 1.0}.get(t, 2), prof)
        assert np.array_equal(test, before)
        assert mask.sum() == 10 * LENGTH
        assert np.array_equal(out[~mask], test[~mask])          # nothing outside the injections
        assert (out != test).any()


def test_one_channel_only_per_site():
    _, test, _, prof, plan = setup("V1")
    out, mask = apply_plan(test, plan, 2, prof)
    for s, c in zip(plan.starts, plan.units):
        changed = (out[s:s + LENGTH] != test[s:s + LENGTH]).any(axis=0)
        assert changed.sum() == 1 and changed[c]


def test_s1_halves_the_period():
    _, test, _, prof, plan = setup("S1")
    out, _ = apply_plan(test, plan, 2, prof)
    s, c = int(plan.starts[0]), plan.units[0]
    assert dominant_lag(out[s:s + LENGTH, c], 0.3, max_lag=25) in (10, 9, 11)   # 20 / 2


def test_s2_quarter_period_breaks_correlation_keeps_partner():
    _, test, _, prof, plan = setup("S2")
    out, _ = apply_plan(test, plan, 0.25, prof)
    for s, (tgt, par) in zip(plan.starts, plan.units):
        sl = slice(int(s), int(s) + LENGTH)
        assert np.array_equal(out[sl, par], test[sl, par])
        assert np.corrcoef(test[sl, 0], test[sl, 1])[0, 1] > 0.9
        assert abs(np.corrcoef(out[sl, 0], out[sl, 1])[0, 1]) < 0.4


def test_s3_full_replacement_uses_same_channel_values_and_keeps_marginal():
    _, test, _, prof, plan = setup("S3")
    out, _ = apply_plan(test, plan, 1.0, prof)
    for s, (tgt, par), d in zip(plan.starts, plan.units, plan.draws):
        sl = slice(int(s), int(s) + LENGTH)
        assert np.array_equal(out[sl, tgt], test[d["src"]:d["src"] + LENGTH, tgt])
        assert abs(d["src"] - s) >= LENGTH
        assert np.array_equal(out[sl, par], test[sl, par])
    half, _ = apply_plan(test, plan, 0.5, prof)
    s = int(plan.starts[0]); tgt = plan.units[0][0]
    assert (half[s:s + LENGTH, tgt] != test[s:s + LENGTH, tgt]).sum() <= 30


def test_volume_magnitudes_exact():
    _, test, _, prof, plan = setup("V1")
    out, _ = apply_plan(test, plan, 2, prof)
    s, c = int(plan.starts[0]), plan.units[0]
    assert np.allclose(out[s:s + LENGTH, c] - test[s:s + LENGTH, c], 2 * prof.std[c])

    _, test, _, prof, plan = setup("V2")
    out, _ = apply_plan(test, plan, 5, prof)
    s, c = int(plan.starts[0]), plan.units[0]
    diff = out[s:s + LENGTH, c] - test[s:s + LENGTH, c]
    assert (diff != 0).sum() == 3 and np.allclose(diff[diff != 0], 5 * prof.std[c])

    _, test, _, prof, plan = setup("V3")
    out, _ = apply_plan(test, plan, 2, prof)
    s, c = int(plan.starts[0]), plan.units[0]
    a, b = test[s:s + LENGTH, c], out[s:s + LENGTH, c]
    assert np.isclose(b.var(), 2 * a.var()) and np.isclose(b.mean(), a.mean())


def test_same_sites_for_all_magnitudes_and_reproducible():
    train, test = toy()
    prof = channel_profile(train)
    lab = np.zeros(len(test), int)
    seed = machine_seed("machine-1-1", "V1")
    assert seed == machine_seed("machine-1-1", "V1") != machine_seed("machine-1-2", "V1")
    p1, p2 = plan_sites(lab, prof, "V1", seed), plan_sites(lab, prof, "V1", seed)
    assert np.array_equal(p1.starts, p2.starts) and p1.units == p2.units
    m1, k1 = apply_plan(test, p1, 1, prof)
    m3, k3 = apply_plan(test, p1, 3, prof)
    assert np.array_equal(k1, k3)


def test_machine_without_eligible_channel_is_skipped_with_reason():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(2000, 4))                       # noise only: nothing periodic or coupled
    prof = channel_profile(train)
    for t in ("S1", "S2", "S3"):
        plan = plan_sites(np.zeros(3000, int), prof, t, 0)
        assert plan.n_placed == 0 and plan.reason == "no eligible channel"
    assert plan_sites(np.zeros(3000, int), prof, "V1", 0).n_placed == 10


def test_too_little_clean_data_places_fewer_and_says_so():
    lab = np.zeros(600, int)
    prof = channel_profile(toy()[0])
    plan = plan_sites(lab, prof, "V1", 0)
    assert 0 < plan.n_placed < 10 and "only" in plan.reason


def test_window_labels_for_injected_series():
    real = np.zeros(300, int)
    real[200:210] = 1
    inj = np.zeros(300, bool)
    inj[60:120] = True
    starts = np.array([0, 60, 100, 130, 170, 200])
    lab = injection_window_labels(real, inj, starts)
    # window 0: injection not in it -> normal; 60: injection fills it -> positive;
    # 100: injection ends at 119 (index 40 of 60) -> final 10 clean -> excluded;
    # 130: clean -> normal; 170: contains real anomaly 200-209 -> excluded; 200: real -> excluded
    assert lab.tolist() == [0, 1, -1, 0, -1, -1]


def test_volume_channels_need_training_std_floor():
    train, _ = toy()
    train[:, 2] *= 0.001                                     # noise channel, std 0.001: below the floor
    prof = channel_profile(train)
    assert 2 not in prof.volume and 3 not in prof.volume and 0 in prof.volume
    _, test = toy()
    plan = plan_sites(np.zeros(len(test), int), prof, "V1", 0)
    assert all(c in prof.volume for c in plan.units)
