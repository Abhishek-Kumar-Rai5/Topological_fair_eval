"""Run metadata is written next to every result."""
import json

import pandas as pd

from src.metadata import run_metadata, save_result


def test_save_result_writes_sidecar(tmp_path):
    meta = run_metadata(seed=3, config={"experiment": "toy"})
    csv, meta_path = save_result(pd.DataFrame({"a": [1, 2]}), tmp_path / "toy.csv", meta)
    saved = json.loads(meta_path.read_text())
    assert csv.exists() and meta_path.name == "toy.meta.json"
    assert saved["seed"] == 3 and "numpy" in saved["libraries"] and "git_commit" in saved
