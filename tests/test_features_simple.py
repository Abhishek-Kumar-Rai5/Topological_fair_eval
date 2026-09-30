"""F1 and F2 features (prereg §4), on toy data."""
import numpy as np
import pytest

from src.feature_sets import FeatureSet
from src.features_simple import (PCAError, correlation_summary, f1_features, f1_names,
                                 f2_names, fft_peak_share, max_autocorrelation)

T = np.arange(60)


def one_window(*channels):
    """Stack 1-D series of length 60 into one window (1, 60, channels)."""
    return np.stack(channels, axis=1)[None]


# ---------------- F1
def test_f1_values():
    w = one_window(np.r_[np.zeros(30), np.ones(30)], np.full(60, 2.0))
    f = f1_features(w)[0]
    # layout: [sum ch0, sum ch1, mean ch0, mean ch1, var ch0, var ch1, unique ch0, unique ch1]
    assert f.tolist() == [30.0, 120.0, 0.5, 2.0, 0.25, 0.0, 2, 1]
    assert len(f1_names(2)) == 8


# ---------------- F2: correlation
def test_correlation_perfectly_coupled():
    x = np.sin(T / 5)
    mean, lo, hi, std, top = correlation_summary(one_window(x, 2 * x + 1, x)[0])
    assert mean == pytest.approx(1) and lo == pytest.approx(1) and std == pytest.approx(0, abs=1e-12)
    assert top == pytest.approx(1)                     # one direction explains everything


def test_correlation_anti_coupled():
    x = np.sin(T / 5)
    mean, lo, hi, std, top = correlation_summary(one_window(x, -x)[0])
    assert lo == pytest.approx(-1) and hi == pytest.approx(-1)


def test_correlation_ignores_constant_channel():
    x = np.sin(T / 5)
    with_const = correlation_summary(one_window(x, 2 * x, np.full(60, 3.0))[0])
    without = correlation_summary(one_window(x, 2 * x)[0])
    assert np.allclose(with_const, without)
    assert correlation_summary(one_window(np.ones(60), np.ones(60))[0]).tolist() == [0] * 5


# ---------------- F2: periodicity
def test_pure_sine_is_periodic_noise_is_not():
    rng = np.random.default_rng(0)
    sine = np.sin(2 * np.pi * T / 10)                  # period 10 -> exact FFT bin
    noise = rng.normal(size=60)
    w = one_window(sine, noise)
    peak = fft_peak_share(w)[0]
    acf = max_autocorrelation(w)[0]
    assert peak[0] == pytest.approx(1.0)               # all power in one frequency
    assert peak[1] < 0.3
    assert acf[0] > 0.7                                 # repeats at lag 10
    assert acf[1] < 0.4


def test_periodicity_constant_channel_is_zero():
    w = one_window(np.full(60, 5.0))
    assert fft_peak_share(w)[0, 0] == 0 and max_autocorrelation(w)[0, 0] == 0


def test_lag_1_is_not_used():
    # a slow trend has huge lag-1 autocorrelation; lags 2-30 must still be < 1
    w = one_window(T.astype(float))
    assert max_autocorrelation(w)[0, 0] < 1


# ---------------- F2: PCA error
def test_pca_error_low_on_normal_high_off_pattern():
    rng = np.random.default_rng(1)
    # normal data: 3 channels all driven by one hidden signal -> lives on a line
    def coupled(n):
        s = rng.normal(size=(n, 60, 1))
        return np.concatenate([s, 2 * s, -s], axis=2) + 0.01 * rng.normal(size=(n, 60, 3))
    train = coupled(50)
    pe = PCAError().fit(train)
    normal_err = pe.transform(coupled(10)).mean()
    broken = coupled(10); broken[:, :, 2] = rng.normal(size=(10, 60))   # channel 2 decouples
    assert pe.transform(broken).mean() > 10 * normal_err


def test_pca_fitted_on_training_only():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(20, 60, 3))
    pe = PCAError().fit(train)
    before = pe.pca.components_.copy()
    pe.transform(rng.normal(size=(5, 60, 3)) * 100)    # transforming must not refit
    assert np.array_equal(before, pe.pca.components_)


# ---------------- feature sets
def test_feature_set_shapes_and_names():
    rng = np.random.default_rng(3)
    train, test = rng.normal(size=(30, 60, 4)), rng.normal(size=(7, 60, 4))
    fs = FeatureSet("F1+F2").fit(train)
    X = fs.transform(test)
    assert X.shape == (7, 4 * 4 + 5 + 2 * 4 + 1)
    assert len(fs.names()) == X.shape[1] and len(f2_names(4)) == 14
    assert np.isfinite(X).all()


def test_unknown_group_rejected():
    with pytest.raises(ValueError):
        FeatureSet("F1+F9")


def test_constant_channel_with_rounding_noise():
    # np.full(60, 0.1).std() is ~1e-17, not 0, because 0.1 is not exact in binary.
    # Such a channel must still count as constant everywhere.
    x = np.sin(T / 5)
    c = np.full(60, 0.1)
    w = one_window(x, 2 * x, c)
    assert np.allclose(correlation_summary(w[0]), correlation_summary(one_window(x, 2 * x)[0]))
    assert fft_peak_share(w)[0, 2] == 0 and max_autocorrelation(w)[0, 2] == 0
