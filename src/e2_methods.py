"""E2: TADA, PHANTOM and TAME (fixed) as detectors on SMD, under OUR rules (prereg §5, §7).

Every adapter has the same three methods, so the runner treats them alike:
    fit(series, size, step)                  fit on the machine's FIT part (first 80% of training), no labels
    score(series, starts, size, advance)     anomaly score per window, higher = more anomalous
    name, deterministic
The runner (not the adapter) sets the threshold: 1% false alarms on VALIDATION scores. Adapters never see labels.
The windows are ours (60 samples, step 10), so every method is scored on the same windows as F1+F2 / F1+F2+F3.

What each adapter changes relative to the released code
-------------------------------------------------------
TADA     `tdaad.TopologicalAnomalyDetector`, default parameters (window 100, step 5, tda_max_dim 1,
         n_centers_by_dim 5), fitted on the fit part. It scores its OWN windows; our window ending at
         e gets the score of the TADA window ending at e (start e - 100), i.e. the same past, never the
         future. The first windows (e < 100) get the first TADA window. Anomaly score = - its robust
         Mahalanobis score. Its own `contamination` threshold is not used.
PHANTOM  The released `Baseline.py` and `AnomalyScore.py`, plus the loop of `main.py` (a script, so
         re-written here): default config (baseline = 4 vectors, anomaly threshold 0.1 only decides
         whether the rolling baseline is updated, Wasserstein distance as main.py calls it with method 2).
         A window is vectorised as per-channel mean, median, min, max (PHANTOM's own strategies).
         Order: baseline = first 4 training windows, stream the rest of the fit windows, then the
         validation windows (scored; state continues), then the test windows from a COPY of that state.
         Deterministic (baseline probability 1), so the 5 seeds give the same numbers.
TAME     TAME's own feature extractor (`utils/tda_features.py`: delay embedding 3x1, VR persistence,
         40x40 persistence images, H0+H1), then StandardScaler + PCA(50) fitted on the training windows,
         then the detector. TWO changes only: our threshold (the runner), and the detector fixed in
         advance to Isolation Forest (200 trees, as in TAME). GMM (4 components) and One-Class SVM
         (nu 0.01, gamma 0.1) are reported separately with TAME's own settings. Deep SVDD needs torch and is
         not run here. Window = ours (60/10), not TAME's 200/200 (see docs/implementation_notes.md).
         TAME has no licence: its code is imported from external/TAME, never copied.
"""
import hashlib
import importlib.util
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.metadata import run_metadata, save_result
from src.metrics import evaluate
from src.smd import load_machine
from src.splits import split_train_val
from src.threshold import threshold_at_fpr
from src.windows import label_windows, window_starts

ROOT = Path(__file__).resolve().parents[1]


def _windows(series, starts, size):
    return np.asarray(series)[np.asarray(starts)[:, None] + np.arange(size)[None, :]]


def _load_file(name, path):
    """Import one file of a third-party repo under a private name (no sys.path pollution)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------ TADA
class TadaAdapter:
    name = "TADA"
    deterministic = False

    def __init__(self, seed, **params):
        self.seed, self.params = seed, params          # empty = tdaad defaults

    def fit(self, series, size, step):
        from tdaad.anomaly_detectors import TopologicalAnomalyDetector
        self.det = TopologicalAnomalyDetector(random_state=self.seed, **self.params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.det.fit(np.asarray(series, float))
        return self

    def score(self, series, starts, size, advance=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self.det._raw_score_samples(np.asarray(series, float))   # one score per TADA window; higher = more normal
        w, st = self.det.window_size, self.det.step
        ends = np.asarray(starts) + size
        idx = np.clip(np.rint((ends - w) / st).astype(int), 0, len(raw) - 1)   # TADA window ending at our end
        return -np.asarray(raw)[idx]


# ------------------------------------------------------------------------------------------ PHANTOM
class PhantomAdapter:
    name = "PHANTOM"
    deterministic = True
    BASELINE_ROWS = 4            # config.json: BASELINE_LENGTH_SECONDS 4000 / WINDOW_LENGTH_SECONDS 1000
    ANOMALY_THRESHOLD = 0.1      # config.json; only decides whether the baseline is updated
    METHOD = 2                   # main.py calls AnomalyScore.compare(barcode, baseBarcode, 2): Wasserstein

    def __init__(self, seed=0, repo=ROOT / "external" / "PHANTOM"):
        self.repo = Path(repo)
        self.base_mod = _load_file("phantom_Baseline", self.repo / "Baseline.py")
        self.score_mod = _load_file("phantom_AnomalyScore", self.repo / "AnomalyScore.py")

    @staticmethod
    def vectorise(windows):
        """Per channel: mean, median, min, max (PHANTOM's mean_values, median_values, min_value, max_value)."""
        w = np.asarray(windows, float)
        return np.concatenate([w.mean(1), np.median(w, 1), w.min(1), w.max(1)], axis=1)

    def _new_baseline(self, vecs):
        import ripser
        b = self.base_mod.Baseline()
        b.vecs = [np.asarray(v, float) for v in vecs]          # Initialize() with probability 1 = the first rows
        b.Compute()
        self.state = {"b": b, "base_barcode": ripser.ripser(b.DMatrix, distance_matrix=True)["dgms"][0]}

    def _step(self, vector, update=True):
        """One iteration of main.py's loop: score the vector, update the rolling baseline if val < threshold."""
        import ripser
        import scipy.spatial
        b = self.state["b"]
        nr = len(b.vecs)
        Y = scipy.spatial.distance.cdist([np.asarray(vector, float)], np.asarray(b.vecs, float),
                                         metric="seuclidean", V=b.var)[0]
        D = np.zeros((nr + 1, nr + 1))
        D[:nr, :nr] = b.DMatrix
        D[nr, :nr] = Y
        D[:nr, nr] = Y
        barcode = ripser.ripser(D, distance_matrix=True)["dgms"][0]
        val = float(self.score_mod.compare(barcode, self.state["base_barcode"], self.METHOD))
        if update and val < self.ANOMALY_THRESHOLD:
            b.vecs.pop(0)
            b.vecs.append(np.asarray(vector, float))
            b.Compute()
            self.state["base_barcode"] = ripser.ripser(b.DMatrix, distance_matrix=True)["dgms"][0]
        return val

    def fit(self, series, size, step):
        vecs = self.vectorise(_windows(series, window_starts(len(series), size, step), size))
        assert len(vecs) > self.BASELINE_ROWS, "not enough training windows for the PHANTOM baseline"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._new_baseline(vecs[:self.BASELINE_ROWS])
            for v in vecs[self.BASELINE_ROWS:]:                # burn-in: the rest of the fit windows
                self._step(v)
        return self

    def score(self, series, starts, size, advance=False):
        """Stream the windows in time order. advance=False works on a copy, so the fitted state is kept."""
        vecs = self.vectorise(_windows(series, starts, size))
        saved = self.state
        if not advance:
            self.state = {"b": deepcopy(saved["b"]), "base_barcode": saved["base_barcode"].copy()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.array([self._step(v) for v in vecs])
        if not advance:
            self.state = saved
        return out


# ------------------------------------------------------------------------------------------ TAME
def tame_feature_fn(repo=ROOT / "external" / "TAME", pi_resolution=40, sigma=1.0):
    """TAME's own persistence-image extractor as a function windows -> features. Needs giotto-tda."""
    mod = _load_file("tame_tda_features", Path(repo) / "utils" / "tda_features.py")
    ext = mod.TDAPersistenceImageExtractor(3, 1, pi_resolution, sigma)     # M_EMBED, TAU, PI_RES, SIGMA of experiment_core
    ext._ensure_gtda()                                                   # fails here, early, if giotto-tda is missing

    def feats(windows):
        cols = [str(i) for i in range(windows.shape[2])]
        return np.stack([ext.transform_one(pd.DataFrame(w, columns=cols), cols) for w in windows])
    return feats


class TameFeatureCache:
    """Compute TAME features once per distinct (series, starts), shared by the IF, GMM and OCSVM adapters."""

    def __init__(self, feature_fn):
        self.feature_fn, self.store = feature_fn, {}

    def get(self, series, starts, size):
        key = (hashlib.blake2b(np.ascontiguousarray(series).tobytes()).hexdigest(),
               np.asarray(starts).tobytes(), size)
        if key not in self.store:
            self.store[key] = np.asarray(self.feature_fn(_windows(series, starts, size)), float)
        return self.store[key]


class TameAdapter:
    deterministic = False
    PCA_K = 50                                   # experiment_core.PCA_K

    def __init__(self, seed, cache, detector="IF"):
        assert detector in ("IF", "GMM", "OCSVM"), detector
        self.seed, self.cache, self.detector = seed, cache, detector
        self.name = f"TAME-{detector}"
        if detector == "OCSVM":
            self.deterministic = True            # one-class SVM has no random state

    def fit(self, series, size, step):
        X = self.cache.get(series, window_starts(len(series), size, step), size)
        self.scaler = StandardScaler().fit(X)                       # training windows only
        k = min(self.PCA_K, len(X) - 1, X.shape[1])
        self.pca = PCA(n_components=k, random_state=42).fit(self.scaler.transform(X))
        Z = self.pca.transform(self.scaler.transform(X))
        if self.detector == "IF":                                   # fixed in advance (prereg §5)
            self.model = IsolationForest(n_estimators=200, random_state=self.seed).fit(Z)
        elif self.detector == "GMM":
            self.model = GaussianMixture(n_components=4, covariance_type="full", random_state=self.seed,
                                         reg_covar=1e-4).fit(Z)
        else:
            self.model = OneClassSVM(kernel="rbf", nu=0.01, gamma=0.1).fit(Z)
        return self

    def score(self, series, starts, size, advance=False):
        X = self.cache.get(series, starts, size)
        return -self.model.score_samples(self.pca.transform(self.scaler.transform(X)))


# ------------------------------------------------------------------------------------------ runner
def build_adapters(cfg, seed, cache=None, feature_fn=None):
    """Adapters for the methods named in cfg['retested_methods'] (TADA, PHANTOM, TAME)."""
    out = []
    for name in cfg["retested_methods"]:
        if name == "TADA":
            out.append(TadaAdapter(seed))
        elif name == "PHANTOM":
            out.append(PhantomAdapter(seed))
        elif name == "TAME":
            for det in ("IF", "GMM", "OCSVM"):
                out.append(TameAdapter(seed, cache, det))
        else:
            raise ValueError(f"unknown method {name}")
    return out


def run_machine(cfg, machine, feature_fn=None):
    """(results, failures) for one machine. A method that raises is recorded, not hidden, and the others still run."""
    w, fpr = cfg["windows"], cfg["threshold"]["false_alarm_rate"]
    size, step, tail = w["size"], w["step"], w["label_tail"]
    train, test, point_labels = load_machine(machine, cfg["data_root"])
    fit_part, val_part = split_train_val(train, cfg["split"]["val_frac"])
    v_starts, t_starts = window_starts(len(val_part), size, step), window_starts(len(test), size, step)
    labels = label_windows(point_labels, t_starts, size, tail)
    cache = None
    if "TAME" in cfg["retested_methods"]:
        cache = TameFeatureCache(feature_fn or tame_feature_fn())
    rows, failures, done = [], [], {}
    for seed in cfg["seeds"]:
        for ad in build_adapters(cfg, seed, cache):
            if ad.name in done and ad.deterministic:                # same numbers for every seed: copy
                rows.append({**done[ad.name], "seed": seed})
                continue
            if ad.name in {f["method"] for f in failures}:
                continue
            try:
                t0 = time.perf_counter()
                ad.fit(fit_part, size, step)
                fit_s = time.perf_counter() - t0
                val_scores = ad.score(val_part, v_starts, size, advance=True)
                thr = threshold_at_fpr(val_scores, fpr)                      # validation only
                t0 = time.perf_counter()
                scores = ad.score(test, t_starts, size)
                per_window = (time.perf_counter() - t0) / len(scores)
                if not np.isfinite(scores).all():
                    raise ValueError("non-finite scores")
                row = {"experiment": cfg["experiment"], "machine": machine, "seed": seed, "feature_set": ad.name,
                       "n_features": np.nan, "threshold": thr, "n_val_windows": len(v_starts),
                       "n_test_windows": len(t_starts), "val_alarm_rate": float((val_scores > thr).mean()),
                       "fit_seconds": fit_s, "runtime_per_window_ms": 1000 * per_window,
                       **evaluate(scores, labels, thr, len(test), cfg.get("minutes_per_point", 1.0))}
                rows.append(row)
                if ad.deterministic:
                    done[ad.name] = row
            except Exception as e:                                           # report the failure, keep going
                failures.append({"machine": machine, "method": ad.name, "seed": seed,
                                 "error": f"{type(e).__name__}: {e}"[:500]})
    return pd.DataFrame(rows), pd.DataFrame(failures, columns=["machine", "method", "seed", "error"])


def run_machine_and_save(cfg, machine, out_dir, feature_fn=None):
    """Save by_machine/<m>_failures.csv, then by_machine/<m>.csv last (it marks the machine as done)."""
    df, fails = run_machine(cfg, machine, feature_fn)
    assert len(df) > 0, f"{machine}: every method failed: {fails.to_dict('records')}"
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"machine": machine})
    out = Path(out_dir) / "by_machine"
    if len(fails):
        save_result(fails, out / f"{machine}_failures.csv", meta)
    save_result(df, out / f"{machine}.csv", meta)
    return df


def merge_results(cfg, machines, out_dir):
    out_dir = Path(out_dir)
    meta = run_metadata(seed=list(cfg["seeds"]), config=cfg, extra={"machines": list(machines)})
    fails = [pd.read_csv(p) for m in machines if (p := out_dir / "by_machine" / f"{m}_failures.csv").exists()]
    if fails:
        save_result(pd.concat(fails, ignore_index=True), out_dir / f"{cfg['experiment']}_failures.csv", meta)
    df = pd.concat([pd.read_csv(out_dir / "by_machine" / f"{m}.csv") for m in machines], ignore_index=True)
    return save_result(df, out_dir / f"{cfg['experiment']}_results.csv", meta)
