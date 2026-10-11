"""E5 runner on toy SMD-format data (no real data)."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import e5
from src.injections import TYPES
from src.metadata import is_valid_result
from src.prepare import prepare_machine

CH = 4


def make_smd(root, n_train=2500, n_test=3000, name="machine-1-1"):
    """Sine channels with period 20 (ch0, ch1 coupled), 24, noise; two labelled level-shift anomalies in test."""
    for sub in ("train", "test", "test_label"):
        Path(root, sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def series(n):
        t = np.arange(n)
        a = np.sin(2 * np.pi * t / 20)
        return 0.5 + 0.2 * np.column_stack([a, a + 0.1 * rng.normal(size=n), np.sin(2 * np.pi * t / 24),
                                            rng.normal(size=n)])
    train, test = series(n_train), series(n_test)
    labels = np.zeros(n_test, int)
    for a in (800, 2000):
        test[a:a + 40] += 0.4
        labels[a:a + 40] = 1
    np.savetxt(Path(root, "train", f"{name}.txt"), train, delimiter=",")
    np.savetxt(Path(root, "test", f"{name}.txt"), test, delimiter=",")
    np.savetxt(Path(root, "test_label", f"{name}.txt"), labels, fmt="%d")
    return train, test, labels


def config(root):
    cfg = yaml.safe_load(Path("configs/e5_injections.yaml").read_text())
    cfg["data_root"] = str(root)
    cfg["seeds"] = [0, 1]
    cfg["detector"]["n_estimators"] = 30
    return cfg


def test_rows_placement_and_files_untouched(tmp_path):
    root = tmp_path / "SMD"
    make_smd(root)
    before = {p: p.read_bytes() for p in root.rglob("*.txt")}
    df, place = e5.run_machine(config(root), "machine-1-1")
    assert {p: p.read_bytes() for p in root.rglob("*.txt")} == before     # data never rewritten
    assert list(place["type"]) == TYPES and (place["n_placed"] == 10).all()
    assert len(df) == 6 * 3 * 2 * 2                      # type x magnitude x feature set x seed
    assert set(df["feature_set"]) == {"F1+F2+F3", "F1+F2"}
    dec = df[df.is_decision]
    assert set(zip(dec["type"], dec["magnitude"])) == {("S1", 2), ("S2", 0.25), ("S3", 1.0),
                                                       ("V1", 2), ("V2", 5), ("V3", 2)}
    assert (df["n_positive"] > 0).all() and (df["n_normal"] > 0).all()
    assert df["pr_auc"].between(0, 1).all()


def test_big_volume_injection_is_detected_by_simple_features(tmp_path):
    root = tmp_path / "SMD"
    make_smd(root)
    df, _ = e5.run_machine(config(root), "machine-1-1")
    # injected PR-AUC well above the chance level (= share of positives) for the strongest V1 shift
    v1 = df[(df["type"] == "V1") & (df.magnitude == 3) & (df.feature_set == "F1+F2")]
    chance = (v1.n_positive / (v1.n_positive + v1.n_normal)).mean()
    assert v1.pr_auc.mean() > 2 * chance


def test_recomputing_only_touched_windows_equals_full_recompute(tmp_path):
    root = tmp_path / "SMD"
    train, test, labels = make_smd(root)
    cfg = config(root)
    mw = prepare_machine("machine-1-1", root=root)
    feats_fn = e5.feature_function(["F1", "F2", "F3"], mw)
    base = feats_fn(mw.test)
    from src.injections import apply_plan, channel_profile, plan_sites
    prof = channel_profile(train[:2000])
    plan = plan_sites(labels, prof, "S2", 3)
    series, mask = apply_plan(test, plan, 0.25, prof)
    fast = e5.injected_test_features(series, mw.test, base, mw.test_starts, mask, feats_fn)
    from src.windows import make_windows
    full = feats_fn(make_windows(series, 60, 10)[0])
    for g in full:
        assert np.allclose(fast[g], full[g]), g
    assert any(not np.allclose(base[g], fast[g]) for g in full)          # something did change


def test_fewer_than_ten_fit_is_reported(tmp_path):
    root = tmp_path / "SMD"
    make_smd(root, n_train=1500, n_test=900)             # too short for 10 injections 180 apart
    df, place = e5.run_machine(config(root), "machine-1-1")
    assert (place["n_placed"] < 10).all() and place["reason"].str.contains("only").all()
    assert (df["n_sites"] < 10).all()


def test_save_merge_and_resume_validity(tmp_path):
    root = tmp_path / "SMD"
    make_smd(root)
    cfg = config(root)
    out = tmp_path / "out"
    e5.run_machine_and_save(cfg, "machine-1-1", out)
    assert is_valid_result(out / "by_machine" / "machine-1-1.csv")
    assert is_valid_result(out / "by_machine" / "machine-1-1_placement.csv")
    path, _ = e5.merge_results(cfg, ["machine-1-1"], out)
    assert is_valid_result(path) and is_valid_result(out / "E5_placement.csv")
    assert not pd.read_csv(path).isna().all().any()
