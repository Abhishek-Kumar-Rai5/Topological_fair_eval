"""Pure helpers for app.py (no Streamlit), so they can be tested.

Everything the app shows comes from files already in the repo (results/, literature/, docs/); nothing is recomputed
except the Live page (src/live.py).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE_COLUMNS = ["id", "year", "title", "cluster", "OUT", "ISO", "USE", "verification", "assessment", "url"]


def read_csv(rel_path):
    """A results/literature CSV as a DataFrame, or None if the file does not exist."""
    p = ROOT / rel_path
    return pd.read_csv(p) if p.exists() else None


def core_papers(master):
    """The 32 core papers: inclusion 'core' (not 'core-method'), D03 excluded (as in literature/make_figs.py)."""
    m = master.astype(str)
    keep = m.inclusion_status.str.startswith("core") & ~m.inclusion_status.str.startswith("core-method") & (m.id != "D03")
    return master[keep][CORE_COLUMNS].reset_index(drop=True)


def per_machine_delta(results, a, b, metric="pr_auc", cluster="machine"):
    """Δ = metric(a) - metric(b) per machine (or scenario), each averaged over seeds. Returns a sorted Series."""
    mean = results.groupby([cluster, "feature_set"])[metric].mean().unstack("feature_set")
    return (mean[a] - mean[b]).sort_values()


def code_audit_text(rel_path="docs/code_verification.md"):
    """The code-audit document without its 'optional email' template."""
    p = ROOT / rel_path
    if not p.exists():
        return None
    return p.read_text().split("## Optional email")[0]


def pooled_base_rate(results):
    """Share of positive windows in a result table (counted once per machine and cell)."""
    keys = [c for c in ("machine", "scenario", "type", "magnitude") if c in results.columns]
    first = results.groupby(keys)[["n_positive", "n_normal"]].first()
    return float(first.n_positive.sum() / (first.n_positive.sum() + first.n_normal.sum()))
