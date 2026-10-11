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


def _atomic_write(path, write):
    """Write with `write(tmp_path)` to a temp file in the same folder, flush it to disk, then rename.

    The rename is atomic, so `path` is either the old complete file or the new complete file,
    never a half-written one (a crash or shutdown leaves only a stray temp file).
    """
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        write(tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _meta_path(path):
    path = Path(path)
    return path.with_name(path.stem + ".meta.json")


def save_result(df, path, meta):
    """Atomically write a DataFrame (.csv or .parquet) plus its <name>.meta.json sidecar.

    Order: data file first, then the sidecar. A result counts as finished only when both
    exist (see is_valid_result), so a crash between the two just means "run it again".
    The sidecar records n_rows, which is_valid_result checks against the data file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        _atomic_write(path, lambda t: df.to_parquet(t, index=False))
    elif path.suffix == ".csv":
        _atomic_write(path, lambda t: df.to_csv(t, index=False))
    else:
        raise ValueError(f"use .csv or .parquet, not {path.suffix}")
    meta = {**meta, "n_rows": len(df)}
    meta_path = _meta_path(path)
    _atomic_write(meta_path, lambda t: t.write_text(json.dumps(meta, indent=2, default=str)))
    return path, meta_path


def is_valid_result(path):
    """True only for a finished result: non-empty data file that parses, plus a readable sidecar.

    Used by --resume. A missing, 0-byte, header-only, truncated or unreadable file, or one whose
    sidecar is missing/corrupt or disagrees on the row count, is NOT valid and will be re-run.
    """
    import pandas as pd
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        meta_path = _meta_path(path)
        if not meta_path.is_file() or meta_path.stat().st_size == 0:
            return False
        meta = json.loads(meta_path.read_text())
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        return len(df) > 0 and meta.get("n_rows", len(df)) == len(df)
    except Exception:                      # any parse error = not valid
        return False
