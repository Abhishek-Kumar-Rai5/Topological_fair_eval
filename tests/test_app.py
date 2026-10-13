"""Smoke test of app.py (Streamlit AppTest) and of the helpers behind it. Toy data for the Live page."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import app_helpers as H
from src import live
from src.features_topo import F3Features
from src.prepare import prepare_machine
from test_runner import make_smd

ROOT = H.ROOT
PAGES = ["Audit", "Code audit", "E2 and E3", "E3x: H0 against H1", "E4: zigzag against its control",
         "E5 and E5b curves", "Live: one window"]


def open_app(page=None):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    if page:
        at.sidebar.radio[0].set_value(page).run()
    return at


def test_default_page_renders():
    at = open_app()
    assert not at.exception
    assert at.header[0].value.startswith("Audit")


@pytest.mark.parametrize("page", PAGES[:-1])
def test_every_results_page_renders(page):
    at = open_app(page)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.header) == 1


def test_live_page_without_data_says_so(monkeypatch, tmp_path):
    monkeypatch.setenv("SMD_ROOT", str(tmp_path / "nothing"))
    at = open_app("Live: one window")
    assert not at.exception
    assert any("not found" in i.value for i in at.info)


def test_live_page_on_toy_data(monkeypatch, tmp_path):
    make_smd(tmp_path / "SMD", names=("machine-1-1", "machine-1-2"))
    monkeypatch.setenv("SMD_ROOT", str(tmp_path / "SMD"))
    at = open_app("Live: one window")
    assert not at.exception, [e.value for e in at.exception]
    assert any("Identical to float32 precision" in s.value for s in at.success)      # MST edges = H0 deaths
    assert len(at.metric) == 2 and {m.label for m in at.metric} == {"F1+F2", "F1+F2+F3"}
    at.radio[0].set_value("positive (anomalous)").run()                                # browse the positive windows
    assert not at.exception


def test_mst_edge_lengths_equal_h0_deaths(tmp_path):
    make_smd(tmp_path / "SMD")
    mw = prepare_machine("machine-1-1", root=tmp_path / "SMD")
    f3 = F3Features().fit(mw.train)
    for i in (0, 5, len(mw.test) - 1):
        v = live.window_view(f3, mw.test[i])
        assert len(v["mst_edges"]) == len(v["h0_deaths"]) > 0
        assert v["max_abs_diff"] < 1e-5
        assert v["cloud3"].shape == (58, 3) and v["points"].shape == (58, 3 * mw.test.shape[2])


def test_scores_use_only_training_fit(tmp_path):
    make_smd(tmp_path / "SMD")
    mw = prepare_machine("machine-1-1", root=tmp_path / "SMD")
    sc = live.fit_scorers(mw)
    out = live.score_window(sc, mw.test[0])
    assert set(out) == {"F1+F2", "F1+F2+F3"}
    for s, thr, alarm in out.values():
        assert np.isfinite(s) and np.isfinite(thr) and alarm == (s > thr)


def test_core_papers_are_32_and_exclude_d03():
    master = H.read_csv("literature/tda_records_master.csv")
    if master is None:
        pytest.skip("literature CSV not in this checkout")
    core = H.core_papers(master)
    assert len(core) == 32 and "D03" not in set(core["id"])


def test_per_machine_delta_and_base_rate():
    df = pd.DataFrame({"machine": ["a", "a", "b", "b"], "feature_set": ["X", "Y"] * 2, "pr_auc": [0.5, 0.4, 0.3, 0.4],
                       "n_positive": [5, 5, 3, 3], "n_normal": [95, 95, 97, 97]})
    d = H.per_machine_delta(df, "X", "Y")
    assert d.to_dict() == pytest.approx({"b": -0.1, "a": 0.1}) and list(d.index) == ["b", "a"]     # sorted
    assert H.pooled_base_rate(df) == pytest.approx(8 / 200)


def test_code_audit_text_drops_the_email_template():
    t = H.code_audit_text()
    if t is None:
        pytest.skip("docs not in this checkout")
    assert "Optional email" not in t and "TAME" in t
