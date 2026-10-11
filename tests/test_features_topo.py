"""F3 topology features (prereg §4), on toy data."""
import numpy as np
import pytest

from src.feature_sets import FeatureSet
from src.features_topo import (BETTI_MULTIPLES, STD_FLOOR, F3Features, betti_curve, delay_embed,
                               diagram_summary, f3_names, persistent_entropy, window_topology,
                               zscore_params)


def test_delay_embedding_3x1():
    w = np.arange(10 * 2).reshape(10, 2)               # 10 steps, 2 channels
    p = delay_embed(w)
    assert p.shape == (8, 6)                            # 10 - 2 points, 3 x 2 dims
    assert p[0].tolist() == w[0].tolist() + w[1].tolist() + w[2].tolist()
    assert delay_embed(np.zeros((60, 38))).shape == (58, 114)


def test_persistent_entropy():
    assert persistent_entropy(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(np.log(4))
    assert persistent_entropy(np.array([2.0])) == 0.0
    assert persistent_entropy(np.array([])) == 0.0


def test_betti_curve_counts_finite_bars_alive():
    dgm = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, np.inf]])
    # alive means birth <= s < death; the infinite bar is dropped
    assert betti_curve(dgm, [0.5, 1.0, 2.0, 5.0]).tolist() == [2, 1, 1, 0]


def test_scales_are_0p2_to_2p0():
    assert BETTI_MULTIPLES.round(6).tolist() == [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    assert "F3_H0_betti_0.2xmed" in f3_names() and "F3_H1_betti_2.0xmed" in f3_names()


def test_zscore_uses_training_stats_and_floors_std():
    rng = np.random.default_rng(0)
    train = np.stack([rng.normal(5, 2, size=(30, 60)), np.full((30, 60), 0.3)], axis=2)  # ch1 constant
    mean, std = zscore_params(train)
    assert mean[0] == pytest.approx(5, abs=0.2) and std[0] == pytest.approx(2, abs=0.2)
    assert mean[1] == pytest.approx(0.3) and std[1] == STD_FLOOR     # no divide by ~0


def test_summary_skips_infinite_bar_everywhere():
    dgm = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, np.inf]])
    s = diagram_summary(dgm, scales=np.array([0.5]))
    n_bars, total, mx, mean, std, ent, betti = s
    assert (n_bars, total, mx, mean) == (2, 4.0, 3.0, 2.0)
    assert std == pytest.approx(1.0)
    assert betti == 2                                    # infinite bar not counted here either


def test_empty_diagram_gives_zeros():
    s = diagram_summary(np.zeros((0, 2)), scales=np.array([0.1, 0.2]))
    assert s.tolist() == [0] * 8


def test_loop_is_found_noise_is_not():
    # two channels tracing a circle once per window -> the embedded points form a loop
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    loop = np.c_[np.cos(t), np.sin(t)]
    noise = np.random.default_rng(0).normal(scale=0.3, size=(60, 2))
    mean, std = np.zeros(2), np.ones(2)
    f_loop = window_topology(loop, mean, std, ref_distance=1.0)
    f_noise = window_topology(noise, mean, std, ref_distance=1.0)
    h1_max = f3_names().index("F3_H1_max_life")
    assert f_loop[h1_max] > 3 * f_noise[h1_max]          # one long-lived H1 bar


def test_constant_window_is_finite():
    f = window_topology(np.full((60, 3), 0.5), np.zeros(3), np.ones(3), ref_distance=1.0)
    names = f3_names()
    assert np.isfinite(f).all()
    assert f[names.index("F3_H0_n_bars")] == 0 and f[names.index("F3_H1_n_bars")] == 0


def test_reference_distance_is_fixed_so_volume_change_is_visible():
    """Scaling a window up must change the features (a per-window scale would hide it)."""
    rng = np.random.default_rng(2)
    train = rng.normal(size=(20, 60, 4))
    f3 = F3Features().fit(train)
    w = rng.normal(size=(60, 4))
    small, big = f3.transform(np.array([w * 0.5, w * 3.0]))
    names = f3_names()
    assert big[names.index("F3_H0_max_life")] > 2 * small[names.index("F3_H0_max_life")]
    assert not np.allclose(small, big)


def test_fit_uses_training_only():
    rng = np.random.default_rng(3)
    train = rng.normal(size=(20, 60, 4))
    a = F3Features().fit(train)
    b = F3Features().fit(train)
    assert a.ref_distance == b.ref_distance and a.ref_distance > 0
    test = rng.normal(loc=10, size=(5, 60, 4))
    a.transform(test)
    assert a.ref_distance == b.ref_distance              # transform never refits


def test_32_features_and_feature_set():
    assert len(f3_names()) == 32
    rng = np.random.default_rng(1)
    train, test = rng.normal(size=(20, 60, 4)), rng.normal(size=(5, 60, 4))
    fs = FeatureSet("F1+F2+F3").fit(train)
    X = fs.transform(test)
    assert X.shape == (5, 16 + 14 + 32) and len(fs.names()) == X.shape[1]
    assert np.isfinite(X).all()
    assert FeatureSet("F1+F3").fit(train).transform(test).shape == (5, 16 + 32)
