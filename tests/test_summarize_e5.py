"""E5 summary on a made-up results table (toy numbers, no real data)."""
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "scripts")
pytest.importorskip("summarize_smd", reason="scripts/summarize_smd.py comes from master; merge master into dev")
import summarize_e5 as s5                                  # noqa: E402

DECISION = {"S1": 2, "S2": 0.25, "S3": 1.0, "V1": 2, "V2": 5, "V3": 2}
MAGS = {"S1": [1.5, 2, 3], "S2": [0.125, 0.25, 0.5], "S3": [0.5, 0.75, 1.0], "V1": [1, 2, 3], "V2": [3, 5, 8], "V3": [1.5, 2, 3]}


def table(gain):
    """gain[type] = PR-AUC(F1+F2+F3) - PR-AUC(F1+F2), the same for every machine (plus small noise)."""
    rng = np.random.default_rng(0)
    rows = []
    for t, mags in MAGS.items():
        for m in mags:
            for mach in range(12):
                base = 0.3 + 0.01 * mach
                for seed in (0, 1):
                    for fs, add in (("F1+F2", 0.0), ("F1+F2+F3", gain[t] + rng.normal(0, 0.002))):
                        rows.append({"machine": f"m{mach}", "type": t, "kind": "shape" if t[0] == "S" else "volume",
                                     "magnitude": m, "is_decision": m == DECISION[t], "seed": seed, "feature_set": fs,
                                     "pr_auc": base + add, "roc_auc": 0.8, "recall_at_thr": 0.5,
                                     "false_alarms_per_day": 1.0, "score_ms_per_window": 1.0, "val_alarm_rate": 0.01,
                                     "n_positive": 40, "n_normal": 400})
    return pd.DataFrame(rows)


def test_p1_supported_p2_supported_when_shape_helps_volume_does_not():
    df = table({"S1": 0.1, "S2": 0.1, "S3": 0.0, "V1": 0.0, "V2": 0.0, "V3": 0.0})
    summary, verdicts = s5.summarize_cells(df, 0)
    assert len(summary) == 6 * 3 * 2 and len(verdicts) == 18
    p = s5.predictions(verdicts).set_index("prediction")["outcome"]
    assert p["P1"] == "Supported" and p["P2"] == "Supported"
    assert verdicts[verdicts.is_decision].shape[0] == 6


def test_p2_refuted_when_a_volume_type_helps_by_at_least_002():
    df = table({"S1": 0.0, "S2": 0.0, "S3": 0.0, "V1": 0.05, "V2": 0.0, "V3": 0.0})
    _, verdicts = s5.summarize_cells(df, 0)
    p = s5.predictions(verdicts).set_index("prediction")["outcome"]
    assert p["P2"] == "Refuted" and p["P1"] == "Refuted"       # shape: 3 x "No added value"


def test_small_volume_gain_is_not_a_refutation():
    df = table({"S1": 0.0, "S2": 0.0, "S3": 0.0, "V1": 0.01, "V2": 0.0, "V3": 0.0})   # helps (CI > 0) but < 0.02
    _, verdicts = s5.summarize_cells(df, 0)
    assert s5.predictions(verdicts).set_index("prediction")["outcome"]["P2"] == "Inconclusive"


def test_curves_figure_written(tmp_path):
    summary, _ = s5.summarize_cells(table({t: 0.0 for t in MAGS}), 0)
    s5.plot_curves(summary, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0
