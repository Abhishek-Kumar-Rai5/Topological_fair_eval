"""Load one experiment config (configs/<exp>.yaml) as a plain dict."""
from pathlib import Path

import yaml


def load_config(path):
    with open(Path(path)) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg
