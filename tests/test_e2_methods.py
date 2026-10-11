"""E2 adapters (TADA, PHANTOM, TAME) on toy data only. No real data, no giotto-tda needed."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src import e2_methods as m
from src.metadata import is_valid_result
from src.windows import window_starts

C, SIZE, STEP = 10, 60, 10


def latent_series(n, rng, broken=()):
    """10 channels = two latent groups of 5 correlated channels (+ noise). Samples in `broken` get independent noise."""
    t = np.arange(n)
    a, b = np.sin(2 * np.pi * t / 37), np.sin(2 * np.pi * t / 23 + 1)
    X = np.column_stack([a] * 5 + [b] * 5) + 0.2 * rng.normal(size=(n, C))
    for lo, hi in broken:
        X[lo:hi] = rng.normal(size=(hi - lo, C))
    return 0.5 + 0.2 * X


def toy(seed=0, n_fit=1500, n_val=400, n_test=1400):
    rng = np.random.default_rng(seed)
    return latent_series(n_fit, rng), latent_series(n_val, rng), latent_series(n_test, rng, broken=[(700, 800)])


def positives(n_test, lo=700, hi=800):
    starts = window_starts(n_test, SIZE, STEP)
    return starts, (starts + SIZE > lo + 50) & (starts + SIZE <= hi + 50)     # windows ending inside / just after the break


def stub_features(windows):
    """Stand-in for TAME's giotto extractor: per-channel mean and std of each window."""
    return np.concatenate([windows.mean(1), windows.std(1)], axis=1)


# ---------- TADA ----------
def test_tada_scores_each_window_and_flags_the_correlation_break():
    fit, val, test = toy()
    ad = m.TadaAdapter(0).fit(fit, SIZE, STEP)
    assert ad.det.window_size == 100 and ad.det.step == 5 and ad.det.n_centers_by_dim == 5   # tdaad defaults
    starts, pos = positives(len(test))
    s = ad.score(test, starts, SIZE)
    assert s.shape == starts.shape and np.isfinite(s).all()
    assert s[pos].mean() > s[~pos].mean()
    assert np.array_equal(s, ad.score(test, starts, SIZE))                   # scoring is repeatable


def test_tada_window_mapping_uses_the_tada_window_ending_where_ours_ends():
    fit, val, test = toy()
    ad = m.TadaAdapter(0).fit(fit, SIZE, STEP)
    raw = ad.det._raw_score_samples(test)
    s = ad.score(test, np.array([0, 10, 140, 300]), SIZE)      # ends 60, 70, 200, 360 -> TADA starts 0 (clipped), 0, 100, 260
    assert np.allclose(s, -raw[[0, 0, 100 // 5, 260 // 5]])


# ---------- PHANTOM ----------
def test_phantom_defaults_and_main_py_choices():
    ad = m.PhantomAdapter()
    assert (ad.BASELINE_ROWS, ad.ANOMALY_THRESHOLD, ad.METHOD) == (4, 0.1, 2)


def test_phantom_vector_and_baseline_start():
    fit, val, test = toy()
    w = fit[:SIZE][None]
    v = m.PhantomAdapter.vectorise(w)
    assert v.shape == (1, 4 * C) and np.allclose(v[0, :C], w[0].mean(0)) and np.allclose(v[0, 3 * C:], w[0].max(0))
    ad = m.PhantomAdapter().fit(fit, SIZE, STEP)
    first4 = m.PhantomAdapter.vectorise(m._windows(fit, window_starts(len(fit), SIZE, STEP)[:4], SIZE))
    assert len(ad.state["b"].vecs) == 4
    # the baseline holds recent fit windows after the burn-in (rolling), never validation or test data
    allv = m.PhantomAdapter.vectorise(m._windows(fit, window_starts(len(fit), SIZE, STEP), SIZE))
    assert all(any(np.allclose(b, a) for a in allv) for b in ad.state["b"].vecs)


def test_phantom_scores_level_shift_higher_and_does_not_touch_state_unless_advance():
    fit, val, test = toy()
    ad = m.PhantomAdapter().fit(fit, SIZE, STEP)
    before = [v.copy() for v in ad.state["b"].vecs]
    starts = window_starts(len(test), SIZE, STEP)
    shifted = test.copy(); shifted[700:800] += 0.5
    s = ad.score(shifted, starts, SIZE)
    assert [np.array_equal(a, b) for a, b in zip(before, ad.state["b"].vecs)] == [True] * 4   # state untouched
    assert np.array_equal(s, ad.score(shifted, starts, SIZE))                                 # deterministic
    pos = (starts + SIZE > 750) & (starts < 800)
    assert s[pos].mean() > s[~pos].mean()
    ad.score(val, window_starts(len(val), SIZE, STEP), SIZE, advance=True)       # advance=True keeps going: state may move
    assert len(ad.state["b"].vecs) == 4                                          # always exactly 4 baseline vectors


def test_phantom_update_rule_only_when_score_below_threshold():
    fit, val, test = toy()
    ad = m.PhantomAdapter().fit(fit, SIZE, STEP)
    v = m.PhantomAdapter.vectorise(m._windows(val, [0], SIZE))[0]
    old = [x.copy() for x in ad.state["b"].vecs]
    far = v + 1000                                                    # huge distance: val >= 0.1, no update
    val_far = ad._step(far)
    assert val_far >= 0.1 and all(np.array_equal(a, b) for a, b in zip(old, ad.state["b"].vecs))
    ad.state["b"].vecs[-1] = v.copy(); ad.state["b"].Compute()        # identical vector: score 0 < 0.1, update
    ad._step(v)
    assert np.array_equal(ad.state["b"].vecs[-1], v)                  # appended (rolled in)


# ---------- TAME ----------
def test_tame_pipeline_fitted_on_training_only_and_fixed_detector():
    fit, val, test = toy()
    cache = m.TameFeatureCache(stub_features)
    ad = m.TameAdapter(3, cache, "IF").fit(fit, SIZE, STEP)
    assert ad.name == "TAME-IF" and ad.model.n_estimators == 200 and ad.model.random_state == 3
    n_train = len(window_starts(len(fit), SIZE, STEP))
    assert ad.scaler.n_samples_seen_ == n_train                       # scaler saw training windows only
    assert ad.pca.n_components_ == min(50, n_train - 1, 2 * C)
    starts, pos = positives(len(test))
    s = ad.score(test, starts, SIZE)
    assert s.shape == starts.shape and np.isfinite(s).all()
    assert ad.scaler.n_samples_seen_ == n_train                       # scoring does not refit anything


def test_tame_other_detectors_have_tame_settings_and_share_features():
    fit, val, test = toy()
    calls = []

    def counting(w):
        calls.append(len(w)); return stub_features(w)
    cache = m.TameFeatureCache(counting)
    ads = [m.TameAdapter(0, cache, d).fit(fit, SIZE, STEP) for d in ("IF", "GMM", "OCSVM")]
    assert len(calls) == 1                                            # training features computed once for all three
    assert ads[1].model.n_components == 4 and ads[1].model.reg_covar == 1e-4
    assert (ads[2].model.nu, ads[2].model.gamma) == (0.01, 0.1) and ads[2].deterministic
    starts = window_starts(len(test), SIZE, STEP)
    for a in ads:
        assert np.isfinite(a.score(test, starts, SIZE)).all()
    assert len(calls) == 2


def test_tame_adapter_has_no_label_or_threshold_input():
    import inspect
    assert list(inspect.signature(m.TameAdapter.score).parameters) == ["self", "series", "starts", "size", "advance"]
    assert not hasattr(m.TameAdapter, "threshold")


def test_tame_real_extractor_if_giotto_installed():
    pytest.importorskip("gtda")
    f = m.tame_feature_fn()
    out = f(toy()[0][None, :SIZE])
    assert out.shape == (1, 2 * 40 * 40)


# ---------- runner ----------
def write_smd(root, name="machine-1-1"):
    for sub in ("train", "test", "test_label"):
        Path(root, sub).mkdir(parents=True, exist_ok=True)
    fit, val, test = toy()
    labels = np.zeros(len(test), int); labels[700:800] = 1
    np.savetxt(Path(root, "train", f"{name}.txt"), np.vstack([fit, val]), delimiter=",")
    np.savetxt(Path(root, "test", f"{name}.txt"), test, delimiter=",")
    np.savetxt(Path(root, "test_label", f"{name}.txt"), labels, fmt="%d")


def cfg(root, methods=("TADA", "PHANTOM", "TAME")):
    c = yaml.safe_load(Path("configs/e2_fair_retest.yaml").read_text())
    c.update(data_root=str(root), seeds=[0, 1], experiment="E2m")
    c["retested_methods"] = {k: v for k, v in c["retested_methods"].items() if k in methods}
    return c


def test_run_machine_rows_threshold_and_deterministic_copies(tmp_path):
    write_smd(tmp_path / "SMD")
    c = cfg(tmp_path / "SMD")
    df, fails = m.run_machine(c, "machine-1-1", feature_fn=stub_features)
    assert len(fails) == 0
    assert set(df.feature_set) == {"TADA", "PHANTOM", "TAME-IF", "TAME-GMM", "TAME-OCSVM"}
    assert len(df) == 5 * 2                                           # every method x 2 seeds
    for name in ("PHANTOM", "TAME-OCSVM"):                            # deterministic: same numbers for each seed
        g = df[df.feature_set == name]
        assert g.pr_auc.nunique() == 1 and set(g.seed) == {0, 1}
    # threshold from validation at 1%: with ~60 val windows at most 1 can alarm; never the test prevalence
    assert (df.val_alarm_rate <= 0.03).all()
    assert (df.n_positive > 0).all() and df.pr_auc.between(0, 1).all()


def test_a_failing_method_is_reported_not_hidden(tmp_path, monkeypatch):
    write_smd(tmp_path / "SMD")
    monkeypatch.setattr(m.TadaAdapter, "fit", lambda self, *a: (_ for _ in ()).throw(RuntimeError("boom")))
    df, fails = m.run_machine(cfg(tmp_path / "SMD", ("TADA", "PHANTOM")), "machine-1-1")
    assert set(df.feature_set) == {"PHANTOM"}
    assert fails.method.tolist() == ["TADA"] and "boom" in fails.error[0]


def test_missing_giotto_fails_early_with_a_clear_error():
    import importlib.util
    if importlib.util.find_spec("gtda") is not None:
        pytest.skip("giotto-tda is installed")
    with pytest.raises(ImportError):
        m.tame_feature_fn()


def test_save_and_merge_are_valid(tmp_path):
    write_smd(tmp_path / "SMD")
    c = cfg(tmp_path / "SMD", ("PHANTOM", "TAME"))
    out = tmp_path / "out"
    m.run_machine_and_save(c, "machine-1-1", out, feature_fn=stub_features)
    assert is_valid_result(out / "by_machine" / "machine-1-1.csv")
    path, _ = m.merge_results(c, ["machine-1-1"], out)
    assert is_valid_result(path) and len(pd.read_csv(path)) == 4 * 2
