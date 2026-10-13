"""CTU-13 graph features, Z-control and Z (zigzag) (prereg §4; notes 54-60).

A window's graph is the bipartite destination graph of src.ctu13.window_graph: nodes are destination
IPs and destination ports, an edge means "this host sent a flow to that IP on that port in the window".

    graph_stats(edges)        V, E, components, cycle rank, degree mean/max/std, diameter
    f2_graph_features(edges)  5 numbers added to F2: components, cycle rank, degree mean, max, std
    z_control(edges)          I03's control, 4 numbers: vertices, edges, components, diameter
    host_zigzag(graphs)       zigzag persistence of one host's window sequence (dionysus)
    zigzag_features(...)      14 numbers per window from the bars, using only times up to that window
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

EXACT_DIAMETER_MAX_NODES = 300
TRAILING_WINDOWS = 5
Z_STATS = ["n_bars", "total", "max", "mean", "std", "entropy", "n_alive"]


# ---------------------------------------------------------------------------------------- graph numbers
def to_index(edges):
    """(E, 2) array of node names -> (n_nodes, int array (E, 2)); empty-safe."""
    if len(edges) == 0:
        return 0, np.empty((0, 2), int)
    names, inv = np.unique(np.asarray(edges, dtype=object).ravel().astype(str), return_inverse=True)
    return len(names), inv.reshape(-1, 2)


def adjacency(n, idx):
    A = sp.coo_matrix((np.ones(len(idx)), (idx[:, 0], idx[:, 1])), shape=(n, n)).tocsr()
    return (A + A.T).minimum(1)


def diameter(A, nodes):
    """Diameter of the component made of `nodes`: exact if small, else a double-sweep lower bound."""
    if len(nodes) <= 1:
        return 0
    sub = A[nodes][:, nodes]
    if len(nodes) <= EXACT_DIAMETER_MAX_NODES:
        d = shortest_path(sub, unweighted=True, directed=False)
        return int(d[np.isfinite(d)].max())
    best, src = 0, 0
    for _ in range(3):                                   # double sweep: farthest node, then again
        d = shortest_path(sub, unweighted=True, directed=False, indices=src)
        d = np.where(np.isfinite(d), d, -1)
        best, src = max(best, int(d.max())), int(d.argmax())
    return best


def graph_stats(edges):
    n, idx = to_index(edges)
    if n == 0:
        return dict(V=0, E=0, components=0, cycle_rank=0, deg_mean=0.0, deg_max=0.0, deg_std=0.0, diameter=0)
    A = adjacency(n, idx)
    n_comp, lab = connected_components(A, directed=False)
    deg = np.asarray(A.sum(axis=1)).ravel()
    biggest = np.flatnonzero(lab == np.bincount(lab).argmax())
    E = int(A.nnz // 2)
    return dict(V=n, E=E, components=int(n_comp), cycle_rank=E - n + int(n_comp), deg_mean=float(deg.mean()),
                deg_max=float(deg.max()), deg_std=float(deg.std()), diameter=diameter(A, biggest))


F2_GRAPH_NAMES = ["g_components", "g_cycle_rank", "g_deg_mean", "g_deg_max", "g_deg_std"]
Z_CONTROL_NAMES = ["zc_vertices", "zc_edges", "zc_components", "zc_diameter"]


def f2_graph_features(edges):
    s = graph_stats(edges)
    return np.array([s["components"], s["cycle_rank"], s["deg_mean"], s["deg_max"], s["deg_std"]], float)


def z_control(edges):
    s = graph_stats(edges)
    return np.array([s["V"], s["E"], s["components"], s["diameter"]], float)


# ---------------------------------------------------------------------------------------- zigzag
def _runs(positions):
    """[2, 3, 4, 7, 8] -> [(2, 4), (7, 8)]  (maximal runs of consecutive window positions)."""
    out, start, prev = [], positions[0], positions[0]
    for p in positions[1:]:
        if p != prev + 1:
            out.append((start, prev))
            start = p
        prev = p
    out.append((start, prev))
    return out


def host_zigzag(graphs):
    """Zigzag persistence of a host's window graphs K_0 ... K_{n-1} (list of (E, 2) name arrays).

    Time index: X_{2i} = K_i, X_{2i+1} = K_i U K_{i+1}. A simplex present in windows i..j (a maximal run)
    is alive on [max(2i-1, 0), 2j+2), and for ever if j is the last window. Returns (dgm0, dgm1) as arrays of
    (birth, death) with death = inf for bars alive at the end, plus n = number of windows.
    """
    import dionysus as d
    n = len(graphs)
    presence = {}                                              # simplex (tuple) -> window positions
    for k, edges in enumerate(graphs):
        for u, v in edges:
            u, v = str(u), str(v)
            for s in ((u,), (v,), tuple(sorted((u, v)))):
                lst = presence.setdefault(s, [])
                if not lst or lst[-1] != k:
                    lst.append(k)
    if not presence:
        return np.empty((0, 2)), np.empty((0, 2)), n
    ids = {name: i for i, name in enumerate(sorted({x for s in presence for x in s}))}
    ordered = sorted(presence, key=lambda s: (len(s), [ids[x] for x in s]))   # vertices before edges
    times = []
    for s in ordered:
        t = []
        for i, j in _runs(presence[s]):
            t.append(max(2 * i - 1, 0))
            if j < n - 1:
                t.append(2 * j + 2)
        times.append(t)
    filt = d.Filtration([d.Simplex([ids[x] for x in s]) for s in ordered])
    assert [tuple(sorted(x)) for x in filt] == [tuple(sorted(ids[v] for v in sm)) for sm in ordered], "filtration order changed"
    _, dgms, _ = d.zigzag_homology_persistence(filt, times)
    arr = [np.array([(p.birth, p.death) for p in dg]).reshape(-1, 2) for dg in dgms[:2]]
    while len(arr) < 2:
        arr.append(np.empty((0, 2)))
    return arr[0], arr[1], n


def _summary(bars, t, trailing):
    """7 numbers for the bars of one dimension seen from window position t (zigzag time 2t)."""
    now = 2 * t
    lo = now - 2 * (trailing - 1)
    if len(bars) == 0:
        return np.zeros(7)
    b, e = bars[:, 0], np.minimum(bars[:, 1], now + 1)             # clip at time <= 2t (no future)
    alive_now = (bars[:, 1] > now) & (b <= now)
    keep = (b <= now) & (e > lo)
    life = (e[keep] - np.maximum(b[keep], lo)).astype(float)
    if len(life) == 0:
        return np.zeros(7)
    p = life / life.sum()
    entropy = float(-(p * np.log(p)).sum())
    return np.array([len(life), life.sum(), life.max(), life.mean(), life.std(), entropy, int(alive_now.sum())], float)


def zigzag_features(dgm0, dgm1, n, trailing=TRAILING_WINDOWS):
    """(n, 14) matrix: 7 summaries of H0 then 7 of H1, for each window position 0..n-1."""
    return np.array([np.concatenate([_summary(dgm0, t, trailing), _summary(dgm1, t, trailing)]) for t in range(n)])


def z_names():
    return [f"z_h{d}_{s}" for d in (0, 1) for s in Z_STATS]
