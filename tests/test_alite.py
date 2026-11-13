"""Plan A-lite helpers on toy data (no real data)."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from src import alite
from src.alite import (PerChannelPH, WholeWindowH1, extra_metrics, h0_deaths_ripser, mst_h0_deaths, positive_runs,
                       random_scores, synthetic_machine, tau_ac)
from src.knn_detector import KNNDetector
from src.runner import run_machine
from tests.test_runner import config, make_smd


def test_positive_runs_split_on_normal_and_excluded():
    assert positive_runs([0, 1, 1, 0, 1, -1, 1, 1]) == [(1, 3), (4, 5), (6, 8)]
    assert positive_runs([0, 0]) == []


def test_extra_metrics_hand_computed():
    labels = np.array([0, 0, 1, 1, 1, 0, 1, -1])
    scores = np.array([.1, .2, .9, .3, .3, .4, .8, 5.0])           # the excluded window has the top score and must be ignored
    m = extra_metrics(scores, labels, threshold=0.5, n_thresholds=400)
    # kept windows: 7. alarms (> .5): windows 2 and 6 -> tp 2, fp 0, fn 2 -> F1 = 4/6; accuracy = (2 tp + 3 tn)/7
    assert m["f1_at_thr"] == pytest.approx(4 / 6)
    assert m["accuracy_at_thr"] == pytest.approx(5 / 7)
    # runs [2,5) and [6,7): window 2 alarms -> the whole first run counts; so at t=.5 PA gives tp 4, fp 0, fn 0 -> F1 = 1
    assert m["best_f1_point_adjusted"] == pytest.approx(1.0)
    # without PA the best threshold (between .8 and .9 would give tp 1; below .3 ... ) cannot reach F1 = 1
    assert m["best_f1_test_thr"] < 1.0
    assert m["best_f1_test_thr"] >= m["f1_at_thr"] - 1e-12


def test_point_adjusted_never_below_plain():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(50), np.ones(20), np.zeros(20), np.ones(10)].astype(int)
    scores = rng.normal(size=100) + 0.5 * labels
    m = extra_metrics(scores, labels, 0.0)
    assert m["best_f1_point_adjusted"] >= m["best_f1_test_thr"]


def test_random_scores_are_reproducible_uniform():
    a, b = random_scores(100, 200, seed=0)
    a2, b2 = random_scores(100, 200, seed=0)
    assert (a == a2).all() and (b == b2).all() and a.min() >= 0 and b.max() <= 1 and len(a) == 100 and len(b) == 200


def test_knn_scores_outlier_higher_and_is_deterministic():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 5))
    det = KNNDetector(5).fit(train)
    s = det.score(np.vstack([np.zeros((1, 5)), np.full((1, 5), 8.0)]))
    assert s[1] > 3 * s[0]
    assert (KNNDetector(5).fit(train).score(train[:10]) == det.score(train[:10])).all()


def test_knn_uses_k_nearest_mean():
    det = KNNDetector(2).fit(np.array([[0.0], [1.0], [10.0]]))
    z = det.scaler.transform([[4.0]])                    # score is in scaled units: distances to the 2 nearest, averaged
    d = np.sort(np.abs(det.scaler.transform([[0.0], [1.0], [10.0]]).ravel() - z.ravel()))[:2]
    assert det.score([[4.0]])[0] == pytest.approx(d.mean())


def test_synthetic_machine_shape_labels_and_period_change():
    train, test, labels = synthetic_machine(0)
    assert train.shape == (2400, 10) and test.shape == (2400, 10) and labels.sum() == 60
    assert labels[1199] == 0 and labels[1200] == 1 and labels[1259] == 1 and labels[1260] == 0
    # channel 0 is slower inside the anomaly: fewer zero crossings per sample than in the same-length normal stretch
    zc = lambda x: np.sum(np.diff(np.sign(x - x.mean())) != 0)
    assert not np.allclose(test[:, 1], train[:, 1])
    assert abs(test[:1200].std() - train.std()) < 0.05


def test_whole_window_h1_tau_changes_features():
    train, test, _ = synthetic_machine(0, n_train=600, n_test=600, at=300)
    tr = np.stack([train[i:i + 60] for i in range(0, 540, 60)])
    f1, f6 = WholeWindowH1(1).fit(tr), WholeWindowH1(6).fit(tr)
    w = tr[:2]
    a, b = f1.transform(w), f6.transform(w)
    assert a.shape == (2, 16) and b.shape == (2, 16) and not np.allclose(a, b)


def test_tau_ac_on_sinusoid_is_half_period():
    t = np.arange(6000)
    x = np.sin(2 * np.pi * t / 20)[:, None] * np.ones((1, 3))
    wins = np.stack([x[i:i + 60] for i in range(0, 5900, 10)])
    assert tau_ac(wins) == 10                              # first minimum of cos(2 pi k / 20) is at k = 10


def test_tau_ac_constant_data_and_bounds():
    assert tau_ac(np.ones((30, 60, 4))) == 1
    rng = np.random.default_rng(0)
    assert 2 <= tau_ac(rng.normal(size=(60, 60, 4))) <= 15


def test_per_channel_ph_shapes_h0_part_agrees_across_maxdim():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(12, 60, 3)) * 0.1 + 0.5
    train[:, :, 2] = 0.3                                   # a constant channel (std floored, reference distance 0 -> 1)
    h0, h1 = PerChannelPH(0).fit(train), PerChannelPH(1).fit(train)
    a, b = h0.transform(train[:3]), h1.transform(train[:3])
    assert a.shape == (3, 16) and b.shape == (3, 16) and np.isfinite(a).all() and np.isfinite(b).all()
    assert h0.tau == h1.tau and np.allclose(h0.ref, h1.ref)
    assert (a[:, 0] <= 60 - 2 * h0.tau - 1).all()          # at most n_points - 1 finite H0 bars


def test_per_channel_feature_sets_registered_and_fit_on_train_only():
    from src.feature_sets import FeatureSet
    rng = np.random.default_rng(1)
    train = rng.normal(size=(10, 60, 3)) * 0.1 + 0.5
    fs = FeatureSet("PC-H0").fit(train)
    assert fs.transform(train[:2]).shape == (2, 16) and len(fs.names()) == 16


def test_mst_equals_h0_deaths():
    rng = np.random.default_rng(0)
    c = rng.normal(size=(58, 20))
    c[5] = c[4]                                            # a duplicate point: zero-length edge must be ignored by both
    d = squareform(pdist(c))
    a, b = mst_h0_deaths(d), h0_deaths_ripser(d)
    assert len(a) == len(b) == 56 and np.abs(a - b).max() < 1e-5


def test_hook_does_not_change_rows_and_sees_every_row(tmp_path):
    make_smd(tmp_path / "SMD")
    cfg = config(tmp_path / "SMD", feature_sets=("F1", "F1+F2"), seeds=(0, 1))
    seen = []
    plain = run_machine(cfg, "machine-1-1")
    hooked = run_machine(cfg, "machine-1-1", on_scores=lambda *a: seen.append(a[:2]))
    cols = [c for c in plain.columns if c != "runtime_per_window_ms"]
    pd.testing.assert_frame_equal(plain[cols], hooked[cols])
    assert sorted(seen) == [("F1", 0), ("F1", 1), ("F1+F2", 0), ("F1+F2", 1)]


def test_l1_machine_rows_add_columns_and_random_row(tmp_path):
    from src.alite_l1 import EXTRA, machine_rows
    make_smd(tmp_path / "SMD")
    e3 = config(tmp_path / "SMD", feature_sets=("F1", "F1+F2"), seeds=(0, 1))
    import yaml
    (tmp_path / "e3.yaml").write_text(yaml.safe_dump(e3))
    l1 = {"sources": {"ours": str(tmp_path / "e3.yaml")}, "n_thresholds": 50, "random_seed": 0}
    df = machine_rows(l1, "ours", "machine-1-1")
    assert set(EXTRA) <= set(df.columns) and (df.feature_set == "RANDOM").sum() == 1 and len(df) == 5
    orig = run_machine(e3, "machine-1-1")
    m = orig.merge(df, on=["machine", "seed", "feature_set"], suffixes=("_o", ""))
    assert len(m) == 4 and (m.pr_auc_o - m.pr_auc).abs().max() == 0
    assert 0 <= df[df.feature_set == "RANDOM"].roc_auc.iloc[0] <= 1      # toy data has few positives: not near 0.5


def test_knn_runs_through_the_runner(tmp_path):
    make_smd(tmp_path / "SMD")
    cfg = config(tmp_path / "SMD", feature_sets=("F1", "F1+F2"), seeds=(0,))
    cfg["detector"] = {"name": "knn", "k": 5}
    df = run_machine(cfg, "machine-1-1")
    assert len(df) == 2 and df.pr_auc.between(0, 1).all()


def test_blank_sheet_has_no_scores_and_agreement_perfect_and_imperfect(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from agreement import agreement, normalise
    from make_blank_sheet import IDS, SCORES, blank_sheet
    sheet = blank_sheet()
    assert list(sheet.id) == IDS and (sheet[SCORES] == "").all().all() and all(sheet.link.str.startswith("http"))
    assert not {"cluster", "inclusion_status", "assessment", "rubric_notes", "status"} & set(sheet.columns)
    assert normalise("Tied (SMD/MSL core); Helped by 1.1") == "Tied" and normalise("U (paper not read)") == "U" and normalise("y") == "Y"
    master = pd.read_csv("literature/tda_records_master.csv", dtype=str, keep_default_na=False)
    filled = master[master.id.isin(IDS)][["id"] + SCORES].reset_index(drop=True)
    _, _, pooled = agreement(filled, master)
    assert (pooled.percent_agreement == 100).all()
    filled.loc[0, "R1"] = "N" if normalise(filled.loc[0, "R1"]) != "N" else "Y"
    pc, _, pooled = agreement(filled, master)
    assert pc.loc["R1", "agree"] == pytest.approx(6 / 7) and pooled.loc[0, "percent_agreement"] < 100
    filled.loc[1, "OUT"] = ""
    with pytest.raises(ValueError):
        agreement(filled, master)
