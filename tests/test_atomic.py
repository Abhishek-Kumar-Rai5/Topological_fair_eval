"""Atomic result writes and the validity check used by --resume (toy data only)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src import metadata
from src.metadata import is_valid_result, save_result

DF = pd.DataFrame({"machine": ["a", "b"], "pr_auc": [0.1, 0.2]})


def test_roundtrip_has_meta_with_row_count(tmp_path):
    p, m = save_result(DF, tmp_path / "x.csv", {"seed": 0})
    assert is_valid_result(p) and json.loads(m.read_text())["n_rows"] == 2
    p2, _ = save_result(DF, tmp_path / "x.parquet", {"seed": 0})
    assert is_valid_result(p2)


def test_no_temp_files_left_behind(tmp_path):
    save_result(DF, tmp_path / "x.csv", {})
    assert sorted(f.name for f in tmp_path.iterdir()) == ["x.csv", "x.meta.json"]


def test_crash_during_write_keeps_old_file_and_leaves_no_temp(tmp_path, monkeypatch):
    p, _ = save_result(DF, tmp_path / "x.csv", {})
    old = p.read_text()

    def boom(self, *a, **k):                       # the write dies half way
        Path(a[0]).write_text("machine,pr_a")
        raise OSError("disk full")
    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    with pytest.raises(OSError):
        save_result(DF.assign(pr_auc=9.0), p, {})
    assert p.read_text() == old and is_valid_result(p)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["x.csv", "x.meta.json"]


def test_missing_empty_or_header_only_is_invalid(tmp_path):
    assert not is_valid_result(tmp_path / "none.csv")
    p, _ = save_result(DF, tmp_path / "a.csv", {})
    p.write_text("")                               # 0 bytes
    assert not is_valid_result(p)
    p, m = save_result(DF, tmp_path / "b.csv", {})
    p.write_text("machine,pr_auc\n")               # header only
    assert not is_valid_result(p)
    p, m = save_result(DF, tmp_path / "c.csv", {})
    m.write_text("")                               # empty sidecar
    assert not is_valid_result(p)
    p, m = save_result(DF, tmp_path / "d.csv", {})
    m.write_text("{not json")                      # corrupt sidecar
    assert not is_valid_result(p)
    p, m = save_result(DF, tmp_path / "e.csv", {})
    m.unlink()                                     # data without sidecar = crash between the two writes
    assert not is_valid_result(p)


def test_truncated_file_disagrees_with_row_count(tmp_path):
    p, _ = save_result(DF, tmp_path / "t.csv", {})
    p.write_text("machine,pr_auc\na,0.1\n")        # one row instead of two
    assert not is_valid_result(p)
    q, _ = save_result(DF, tmp_path / "t.parquet", {})
    q.write_bytes(q.read_bytes()[:20])             # cut parquet
    assert not is_valid_result(q)


def test_resume_filter_in_script(tmp_path):
    """The script's resume test: only valid files are skipped."""
    d = tmp_path / "by_machine"
    save_result(DF, d / "m1.csv", {})
    (d / "m2.csv").write_text("")
    (d / "m3.csv").write_text("machine,pr_auc\n")
    machines = ["m1", "m2", "m3", "m4"]
    todo = [m for m in machines if not is_valid_result(d / f"{m}.csv")]
    assert todo == ["m2", "m3", "m4"]
