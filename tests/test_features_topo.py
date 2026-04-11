"""F3 topology features (prereg §4), on toy data."""
import numpy as np
import pytest

from src.feature_sets import FeatureSet
from src.features_topo import (betti_curve, delay_embed, diagram_summary, f3_names,
                               persistent_entropy, window_topology)


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


def test_betti_curve_counts_bars_alive():
    dgm = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, np.inf]])
    # alive means birth <= s < death
    assert betti_curve(dgm, [0.5, 1.0, 2.0, 5.0]).tolist() == [3, 2, 2, 1]


def test_summary_skips_infinite_bar_in_stats_but_not_in_betti():
    dgm = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, np.inf]])
    s = diagram_summary(dgm, scales=np.array([0.5]))
    n_bars, total, mx, mean, std, ent, betti = s
    assert (n_bars, total, mx, mean) == (2, 4.0, 3.0, 2.0)
    assert std == pytest.approx(1.0)
    assert betti == 3                                    # infinite bar counted here


def test_empty_diagram_gives_zeros():
    s = diagram_summary(np.zeros((0, 2)), scales=np.array([0.1, 0.2]))
    assert s.tolist() == [0] * 8


def test_loop_is_found_noise_is_not():
    # two channels tracing a circle once per window -> the embedded points form a loop
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    loop = np.c_[np.cos(t), np.sin(t)]
    noise = np.random.default_rng(0).normal(scale=0.3, size=(60, 2))
    f_loop, f_noise = window_topology(loop), window_topology(noise)
    names = f3_names()
    h1_max = names.index("F3_H1_max_life")
    assert f_loop[h1_max] > 3 * f_noise[h1_max]          # one long-lived H1 bar


def test_constant_window_is_finite():
    f = window_topology(np.full((60, 3), 0.5))
    names = f3_names()
    assert np.isfinite(f).all()
    assert f[names.index("F3_H0_betti_1.0xmed")] == 1    # one component
    assert f[names.index("F3_H1_n_bars")] == 0           # no loops


def test_32_features_and_feature_set():
    assert len(f3_names()) == 32
    rng = np.random.default_rng(1)
    train, test = rng.normal(size=(20, 60, 4)), rng.normal(size=(5, 60, 4))
    fs = FeatureSet("F1+F2+F3").fit(train)
    X = fs.transform(test)
    assert X.shape == (5, 16 + 14 + 32) and len(fs.names()) == X.shape[1]
    assert np.isfinite(X).all()
    assert FeatureSet("F1+F3").fit(train).transform(test).shape == (5, 16 + 32)
