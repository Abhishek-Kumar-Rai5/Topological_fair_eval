"""Runner on toy SMD-format data (no real data needed)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.feature_sets import FeatureSet
from src.prepare import prepare_machine
from src.runner import group_features, merge_results, pipeline_spec, run_machine, run_machine_and_save

CHANNELS, N = 4, 700


def make_smd(root, names=("machine-1-1", "machine-1-2"), flip_labels=False):
    """Toy SMD folder: sinusoids + noise; the test series has two level-shift anomalies."""
    for sub in ("train", "test", "test_label"):
        Path(root, sub).mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(names):
        rng = np.random.default_rng(k)
        t = np.arange(N)[:, None]
        base = lambda: 0.5 + 0.2 * np.sin(2 * np.pi * t / (20 + 5 * np.arange(CHANNELS))) + rng.normal(0, 0.03, (N, CHANNELS))
        train, test = base(), base()
        labels = np.zeros(N, dtype=int)
        for a in (200, 450):
            test[a:a + 40] += 0.4
            labels[a:a + 40] = 1
        if flip_labels:
            labels = 1 - labels
        np.savetxt(Path(root, "train", f"{name}.txt"), train, delimiter=",")
        np.savetxt(Path(root, "test", f"{name}.txt"), test, delimiter=",")
        np.savetxt(Path(root, "test_label", f"{name}.txt"), labels, fmt="%d")


def config(root, feature_sets=("F1", "F1+F2"), seeds=(0, 1)):
    return {"experiment": "T3", "dataset": "SMD", "data_root": str(root),
            "windows": {"size": 60, "step": 10, "label_tail": 10}, "split": {"val_frac": 0.2},
            "threshold": {"false_alarm_rate": 0.01, "on": "validation"}, "seeds": list(seeds),
            "minutes_per_point": 1.0,
            "detector": {"name": "isolation_forest", "n_estimators": 50, "max_samples": "auto"},
            "feature_sets": list(feature_sets)}


def test_rows_columns_and_validation_rate(tmp_path):
    make_smd(tmp_path / "SMD")
    df = run_machine(config(tmp_path / "SMD"), "machine-1-1")
    assert len(df) == 2 * 2                               # 2 feature sets x 2 seeds
    for col in ("pr_auc", "roc_auc", "recall_at_thr", "false_alarms_per_day", "threshold",
                "runtime_per_window_ms", "n_positive", "n_normal", "feature_set", "seed"):
        assert col in df.columns
    assert (df["val_alarm_rate"] <= 0.01).all()           # threshold is set on validation at 1%
    assert df["pr_auc"].between(0, 1).all() and df["n_positive"].min() > 0


def test_same_seed_same_result(tmp_path):
    make_smd(tmp_path / "SMD")
    a = run_machine(config(tmp_path / "SMD"), "machine-1-1").drop(columns="runtime_per_window_ms")
    b = run_machine(config(tmp_path / "SMD"), "machine-1-1").drop(columns="runtime_per_window_ms")
    pd.testing.assert_frame_equal(a, b)


def test_test_labels_do_not_change_scores_or_threshold(tmp_path):
    """Flipping every test label must change the metrics but never the threshold."""
    make_smd(tmp_path / "A")
    make_smd(tmp_path / "B", flip_labels=True)
    a = run_machine(config(tmp_path / "A"), "machine-1-1")
    b = run_machine(config(tmp_path / "B"), "machine-1-1")
    assert a["threshold"].tolist() == b["threshold"].tolist()
    assert a["val_alarm_rate"].tolist() == b["val_alarm_rate"].tolist()
    assert not np.allclose(a["pr_auc"], b["pr_auc"])


def test_group_features_match_feature_set(tmp_path):
    make_smd(tmp_path / "SMD")
    mw = prepare_machine("machine-1-1", root=tmp_path / "SMD")
    feats = group_features(mw, ["F1", "F2", "F3"])
    fs = FeatureSet("F1+F2+F3").fit(mw.train)
    X_test = np.hstack([feats[g]["test"] for g in ("F1", "F2", "F3")])
    assert np.allclose(X_test, fs.transform(mw.test))
    assert feats["F3"]["seconds_per_window"] > 0


def test_save_merge_and_metadata(tmp_path):
    make_smd(tmp_path / "SMD")
    cfg, out = config(tmp_path / "SMD"), tmp_path / "res"
    for m in ("machine-1-1", "machine-1-2"):
        run_machine_and_save(cfg, m, out)
    path, meta_path = merge_results(cfg, ["machine-1-1", "machine-1-2"], out)
    df = pd.read_csv(path)
    assert len(df) == 2 * 2 * 2 and set(df["machine"]) == {"machine-1-1", "machine-1-2"}
    meta = json.loads(meta_path.read_text())
    assert meta["seed"] == [0, 1] and meta["git_commit"] and "scikit-learn" in meta["libraries"]
    assert (out / "by_machine" / "machine-1-1.meta.json").exists()


def test_pipeline_spec():
    assert pipeline_spec({"detector": {"name": "isolation_forest"}, "feature_sets": ["F1"]})[1] == ["F1"]
    e2 = {"our_pipelines": {"detector": {"name": "isolation_forest"}, "feature_sets": ["F1+F2"]}}
    assert pipeline_spec(e2)[1] == ["F1+F2"]
    with pytest.raises(ValueError):
        pipeline_spec({"experiment": "E5", "injection": {}, "feature_sets": ["F1"]})
    with pytest.raises(ValueError):
        pipeline_spec({"experiment": "E1", "method": {}})
