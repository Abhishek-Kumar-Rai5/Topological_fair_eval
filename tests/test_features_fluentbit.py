"""F5 Fluent Bit-style features, checked against small worked examples (hand-computed)."""
import numpy as np
import pytest

import src.features_fluentbit as fb
from src.feature_sets import FeatureSet


def test_signed_log1p_of_first_differences_like_tda_c():
    w = np.array([[0.0], [1.0], [3.0], [2.0]])           # diffs 1, 2, -1
    v = fb.signed_log1p_rate(w)                          # dt = 1 sample
    assert v.shape == (3, 1)                             # one row lost: the first sample only gives a baseline
    assert v[:, 0].tolist() == pytest.approx([np.log(2), np.log(3), -np.log(2)])
    assert fb.signed_log1p_rate(w, dt=2.0)[0, 0] == pytest.approx(np.log1p(0.5))   # rate = diff / dt
    assert fb.signed_log1p_rate(np.ones((3, 2))).tolist() == [[0, 0], [0, 0]]       # constant -> 0


def test_delay_embedding_3x1_stacks_three_latest_vectors():
    v = np.arange(6, dtype=float).reshape(6, 1)          # 0..5
    d = fb.delay_distances(v)                            # 4 points: (2,1,0) (3,2,1) (4,3,2) (5,4,3)
    assert d.shape == (4, 4) and d.dtype == np.float32
    assert d[1, 0] == pytest.approx(np.sqrt(3))          # each coordinate differs by 1
    assert d[3, 0] == pytest.approx(3 * np.sqrt(3))


def test_quantile_threshold_index_is_floor_q_times_k_minus_1():
    pts = np.array([[0.0], [1.0], [3.0]])                # distances 1, 2, 3 -> k = 3
    d = fb.delay_distances(pts, m=1)
    assert float(fb.quantile_threshold(d, 0.1)) == 1.0   # floor(0.2) = 0
    assert float(fb.quantile_threshold(d, 0.5)) == 2.0   # floor(1.0) = 1
    assert float(fb.quantile_threshold(d, 0.9)) == 2.0   # floor(1.8) = 1, no interpolation


SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


def test_worked_example_unit_square():
    """Square, side 1 (diagonal sqrt 2). Distances sorted: 1,1,1,1,1.414,1.414.

    q = 0.1..0.7 -> threshold 1: three finite H0 bars, the loop is still alive at the
        threshold, so it is NOT counted: beta = (3, 0, 0).
    q = 0.8, 0.9 -> threshold 1.414: the loop (born 1, dies 1.414) is finite: (3, 1, 0).
    """
    d = fb.delay_distances(SQUARE, m=1)
    assert fb.count_bars(d, fb.quantile_threshold(d, 0.5)) == ([3, 0, 0], 1)
    assert fb.count_bars(d, fb.quantile_threshold(d, 0.9)) == ([3, 1, 0], 2)


def test_alive_bars_at_threshold_are_not_counted():
    d = fb.delay_distances(SQUARE, m=1)
    betti, _ = fb.count_bars(d, 1.0)
    assert betti[1] == 0                                 # a loop exists at 1.0 but has no finite death
    # without a threshold the loop is finite (dies at 1.414) and the same as above
    assert fb.count_bars(d, 0.0)[0] == [3, 1, 0]         # threshold <= 0 means no limit


def test_bars_shorter_than_one_thousandth_are_ignored():
    pts = np.array([[0.0], [0.0005], [5.0]])             # a 0.0005 merge, then a 4.9995 merge
    d = fb.delay_distances(pts, m=1)
    assert fb.count_bars(d, 0.0)[0][0] == 1              # only the long H0 bar counts


def test_selection_rules(monkeypatch):
    """Rule 1: a scale with more H1 bars wins. Rule 2: if no H1 yet, the larger beta0 wins."""
    monkeypatch.setattr(fb, "signed_log1p_rate", lambda w, dt=1.0: np.zeros((20, 2)))
    monkeypatch.setattr(fb, "delay_distances", lambda v: np.zeros((18, 18), dtype=np.float32))
    monkeypatch.setattr(fb, "quantile_threshold", lambda d, q: q)   # threshold = q, so we can script it
    scripted = {0.1: ([5, 0, 0], 1), 0.2: ([7, 0, 1], 3), 0.3: ([6, 2, 0], 2), 0.4: ([9, 2, 4], 3),
                0.5: ([8, 3, 1], 3), 0.6: ([1, 0, 0], 1), 0.7: ([1, 0, 0], 1), 0.8: ([1, 0, 0], 1),
                0.9: ([1, 0, 0], 1)}
    monkeypatch.setattr(fb, "count_bars", lambda d, thr: scripted[thr])
    # q0.1: b0=5 (rule 2). q0.2: b1=0, b0=7>5 -> b0=7,b2=1 (rule 2). q0.3: b1=2>0 -> (6,2,0).
    # q0.4: b1=2 not > 2, best_b1 != 0 -> nothing. q0.5: b1=3>2 -> (8,3,1). The rest: nothing.
    assert fb.window_betti(np.zeros((21, 2))).tolist() == [8, 3, 1]


def test_constant_window_gives_zeros_and_short_window_zeros():
    assert fb.window_betti(np.full((60, 5), 0.3)).tolist() == [0, 0, 0]
    assert fb.window_betti(np.zeros((10, 3))).tolist() == [0, 0, 0]   # 9 vectors < min_points (10)


def test_loop_gives_beta1():
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    w = np.c_[np.sin(t), np.cos(t)]
    assert fb.window_betti(w)[1] >= 1                    # a circle in two channels: a loop is seen


def test_feature_set_f5():
    rng = np.random.default_rng(0)
    train, test = rng.normal(size=(5, 60, 4)), rng.normal(size=(3, 60, 4))
    fs = FeatureSet("F5").fit(train)
    X = fs.transform(test)
    assert X.shape == (3, 3) and fs.names() == ["F5_beta0", "F5_beta1", "F5_beta2"]
    assert np.isfinite(X).all()
