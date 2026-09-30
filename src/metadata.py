"""Run metadata saved next to every result file (CLAUDE.md hard rule 5).

Every result `results/<exp>/<name>.csv` (or .parquet) gets a sidecar
`results/<exp>/<name>.meta.json` with: git commit (and whether the working tree
had uncommitted changes), seed, config, Python and library versions, hardware, time.
"""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

LIBRARIES = ["numpy", "pandas", "scipy", "scikit-learn", "ripser", "persim", "gudhi",
             "dionysus", "tdaad", "pyarrow", "PyYAML"]


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def library_versions():
    out = {}
    for lib in LIBRARIES:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = None
    return out


def run_metadata(seed, config=None, extra=None):
    """Everything needed to reproduce one run, as a JSON-friendly dict."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,  # True = uncommitted changes
        "seed": seed,
        "config": config,
        "python": sys.version.split()[0],
        "libraries": library_versions(),
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }


def save_result(df, path, meta):
    """Write a DataFrame (.csv or .parquet) plus its <name>.meta.json sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"use .csv or .parquet, not {path.suffix}")
    meta_path = path.with_name(path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return path, meta_path
