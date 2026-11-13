"""Live page of the app: the topology of one SMD window, and its F1+F2 / F1+F2+F3 scores (no Streamlit here).

Everything is fitted on the machine's TRAINING windows only (the same rules as E3). Labels are never used by any
function in this file; the app shows a window's label after scoring, for display.
"""
import numpy as np
from ripser import ripser
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.decomposition import PCA

from src.detector import IFDetector
from src.feature_sets import GROUPS
from src.features_topo import delay_embed, embedded_distances
from src.threshold import threshold_at_fpr

SETS = {"F1+F2": ("F1", "F2"), "F1+F2+F3": ("F1", "F2", "F3")}
ZERO = 1e-9                                  # distances below this are "the same point" (ripser reports no bar for them)


def window_view(f3, window):
    """The F3 pipeline on one window, step by step.

    f3: a fitted src.features_topo.F3Features. Returns a dict with
        points  (58, 114) z-scored delay-embedded cloud      cloud3  (58, 3) its PCA projection (+ explained share)
        dgm0, dgm1  ripser H0 / H1 diagrams                  dist    (58, 58) Euclidean distances (float32)
        mst_edges, h0_deaths  sorted edge lengths of the minimum spanning tree, sorted finite H0 death times
        max_abs_diff  largest |MST edge - H0 death| (None if the counts differ)
    """
    window = np.asarray(window, float)
    points = delay_embed((window - f3.mean) / f3.std)
    dist = embedded_distances(window, f3.mean, f3.std)
    pca = PCA(n_components=3, random_state=0)
    dgm0, dgm1 = ripser(dist, maxdim=1, distance_matrix=True)["dgms"]
    d = dist.astype(float)
    np.fill_diagonal(d, 0.0)
    off = ~np.eye(len(d), dtype=bool)
    d[off & (d == 0)] = ZERO / 10                      # scipy treats a 0 as "no edge"; keep duplicate points connected
    mst = np.sort(minimum_spanning_tree(csr_matrix(d)).data)   # sparse input: a dense one drops values below ~1e-8
    mst = mst[mst > ZERO]
    deaths = np.sort(dgm0[np.isfinite(dgm0[:, 1]), 1])
    deaths = deaths[deaths > ZERO]
    same = len(mst) == len(deaths)
    return {"points": points, "cloud3": pca.fit_transform(points), "explained": pca.explained_variance_ratio_,
            "dgm0": dgm0, "dgm1": dgm1, "dist": dist, "mst_edges": mst, "h0_deaths": deaths,
            "max_abs_diff": float(np.abs(mst - deaths).max()) if same and len(mst) else (0.0 if same else None)}


def fit_scorers(mw, seed=0, fpr=0.01):
    """Fit F1, F2, F3 and one Isolation Forest per feature set on the training windows; threshold on validation."""
    groups = {g: GROUPS[g][0]().fit(mw.train) for g in ("F1", "F2", "F3")}
    train = {g: groups[g].transform(mw.train) for g in groups}
    val = {g: groups[g].transform(mw.val) for g in groups}
    sets = {}
    for name, parts in SETS.items():
        det = IFDetector(seed).fit(np.hstack([train[g] for g in parts]))
        sets[name] = (det, threshold_at_fpr(det.score(np.hstack([val[g] for g in parts])), fpr))
    return {"groups": groups, "sets": sets, "seed": seed, "fpr": fpr}


def score_window(scorers, window):
    """{set name: (score, threshold, alarm)} for one window (60 x channels)."""
    w = np.asarray(window, float)[None]
    feats = {g: scorers["groups"][g].transform(w) for g in scorers["groups"]}
    out = {}
    for name, parts in SETS.items():
        det, thr = scorers["sets"][name]
        s = float(det.score(np.hstack([feats[g] for g in parts]))[0])
        out[name] = (s, thr, s > thr)
    return out
