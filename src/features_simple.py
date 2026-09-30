"""F1 (simple) and F2 (simple-structure) features for SMD windows (prereg §4).

Input everywhere: windows, an array (n_windows, window_size, n_channels).
Output: a 2-D feature matrix (n_windows, n_features), plus feature names.

F1, per channel (4 x 38 = 152 features):
    sum, mean, variance, number of unique values
F2 (3 groups):
    correlation summary (5): mean, min, max, std of the off-diagonal channel
        correlations, and the top-eigenvalue share of the correlation matrix
    periodicity, per channel (2 x 38 = 76): FFT peak share, and the maximum
        autocorrelation at lags 2-30
    PCA reconstruction error (1): PCA fitted on TRAINING points only
F1 has nothing to fit. F2's PCA (and its scaler) is fitted on training windows only.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ACF_LAGS = range(2, 31)          # prereg: lags 2-30
PCA_VARIANCE_KEPT = 0.95         # fixed in advance, no tuning (see docs/implementation_notes.md)


def is_varying(windows, axis):
    """True where a channel is NOT constant over time.

    Uses max - min > 0, which is exact. (std > 0 is not: a constant 0.1 has
    std ~1e-17 because of floating-point rounding.)
    """
    return np.ptp(windows, axis=axis) > 0


# ---------------------------------------------------------------- F1
def f1_features(windows):
    """Per channel: sum, mean, variance, unique-value count. Shape (n, 4 * channels)."""
    w = np.asarray(windows, dtype=float)
    total = w.sum(axis=1)
    mean = w.mean(axis=1)
    var = w.var(axis=1)
    # unique count: sort each channel's values in time, count where the value changes
    s = np.sort(w, axis=1)
    unique = 1 + (np.diff(s, axis=1) != 0).sum(axis=1)
    return np.hstack([total, mean, var, unique])


def f1_names(n_channels):
    return [f"F1_{stat}_ch{c}" for stat in ("sum", "mean", "var", "unique") for c in range(n_channels)]


# ---------------------------------------------------------------- F2: correlation
def correlation_summary(window):
    """5 numbers describing how the channels of ONE window move together.

    Constant channels have no defined correlation, so they are left out.
    If fewer than 2 channels vary, there is no correlation to describe: return zeros.
    """
    varying = is_varying(window, axis=0)
    if varying.sum() < 2:
        return np.zeros(5)
    corr = np.corrcoef(window[:, varying], rowvar=False)
    off_diag = corr[~np.eye(len(corr), dtype=bool)]
    eigvals = np.linalg.eigvalsh(corr)                  # symmetric matrix -> real eigenvalues
    top_share = eigvals.max() / eigvals.sum()           # sum = trace = number of channels
    return np.array([off_diag.mean(), off_diag.min(), off_diag.max(), off_diag.std(), top_share])


# ---------------------------------------------------------------- F2: periodicity
def fft_peak_share(windows):
    """Per channel: share of the signal's power in its single strongest frequency.

    Close to 1 = one dominant rhythm; small = power spread out (noise-like).
    The mean is removed first, so the constant (DC) part does not count.
    Constant channels get 0.
    """
    x = windows - windows.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    power = power[:, 1:, :]                             # drop frequency 0 (DC)
    total = power.sum(axis=1)
    peak = power.max(axis=1)
    ok = is_varying(windows, axis=1)
    return np.divide(peak, total, out=np.zeros_like(peak), where=ok)


def max_autocorrelation(windows, lags=ACF_LAGS):
    """Per channel: the highest autocorrelation over lags 2-30.

    autocorrelation(k) = sum_t x_t * x_(t+k) / sum_t x_t^2, with x = channel minus its mean
    (the standard estimator). High = the channel repeats itself after k steps.
    Constant channels get 0.
    """
    x = windows - windows.mean(axis=1, keepdims=True)
    denom = (x ** 2).sum(axis=1)
    ok = is_varying(windows, axis=1)
    best = np.full(denom.shape, -np.inf)
    for k in lags:
        num = (x[:, :-k, :] * x[:, k:, :]).sum(axis=1)
        best = np.maximum(best, np.divide(num, denom, out=np.zeros_like(num), where=ok))
    return best


# ---------------------------------------------------------------- F2: PCA error
class PCAError:
    """Reconstruction error of each window under a PCA of NORMAL training points.

    fit:   scale each channel and fit PCA on all time points of the training windows,
           keeping enough components for 95% of the training variance
    score: project every point of a window onto those components and back; the window's
           feature is the mean squared reconstruction error over its points
    Points unlike normal training data reconstruct badly -> high error.
    """

    def fit(self, train_windows):
        points = train_windows.reshape(-1, train_windows.shape[2])   # (n * size, channels)
        self.scaler = StandardScaler().fit(points)
        self.pca = PCA(n_components=PCA_VARIANCE_KEPT, svd_solver="full").fit(self.scaler.transform(points))
        return self

    def transform(self, windows):
        n, size, ch = windows.shape
        z = self.scaler.transform(windows.reshape(-1, ch))
        recon = self.pca.inverse_transform(self.pca.transform(z))
        err = ((z - recon) ** 2).mean(axis=1)                        # per point
        return err.reshape(n, size).mean(axis=1, keepdims=True)      # per window


# ---------------------------------------------------------------- F2 together
class F2Features:
    """Correlation summary (5) + periodicity (2 per channel) + PCA error (1)."""

    def fit(self, train_windows):
        self.pca_error = PCAError().fit(np.asarray(train_windows, dtype=float))
        return self

    def transform(self, windows):
        w = np.asarray(windows, dtype=float)
        corr = np.array([correlation_summary(x) for x in w])
        return np.hstack([corr, fft_peak_share(w), max_autocorrelation(w), self.pca_error.transform(w)])


def f2_names(n_channels):
    corr = [f"F2_corr_{s}" for s in ("mean", "min", "max", "std", "top_eig_share")]
    period = [f"F2_{s}_ch{c}" for s in ("fft_peak_share", "max_acf") for c in range(n_channels)]
    return corr + period + ["F2_pca_error"]
