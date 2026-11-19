"""Smoke-test the environment on toy data only (no SMD / CTU-13 access).

Run: .venv/bin/python scripts/check_env.py
Prints the Python and library versions, then checks that each TDA library runs.
"""
import sys
from importlib.metadata import version

import numpy as np

LIBS = ["numpy", "pandas", "scipy", "scikit-learn", "ripser", "persim", "gudhi",
        "dionysus", "tdaad", "matplotlib", "streamlit", "pyarrow", "PyYAML", "pytest"]

print("python", sys.version.split()[0])
assert sys.version_info[:2] == (3, 11), "expected Python 3.11"
for lib in LIBS:
    print(f"  {lib:13s} {version(lib)}")

rng = np.random.default_rng(0)

# A noisy circle should have one long H1 bar.
t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
circle = np.c_[np.cos(t), np.sin(t)] + 0.05 * rng.normal(size=(60, 2))

from ripser import ripser
h1 = ripser(circle, maxdim=1)["dgms"][1]
longest = float((h1[:, 1] - h1[:, 0]).max())
assert longest > 1.0, longest
print("ripser   ok: longest H1 bar", round(longest, 3))

import persim
assert persim.bottleneck(h1, h1) == 0.0
print("persim   ok")

import gudhi
st = gudhi.RipsComplex(points=circle, max_edge_length=1.0).create_simplex_tree(max_dimension=2)
print("gudhi    ok:", st.num_simplices(), "simplices")

import dionysus as d
f = d.Filtration([d.Simplex([0]), d.Simplex([1]), d.Simplex([0, 1])])
times = [[0], [0], [1, 2]]  # edge appears at 1, disappears at 2
_, dgms, _ = d.zigzag_homology_persistence(f, times)
print("dionysus ok: zigzag H0 bars", len(dgms[0]))

# tdaad uses sensors as points, so it needs many channels (SMD has 38).
from tdaad.anomaly_detectors import TopologicalAnomalyDetector
X = rng.normal(size=(600, 38))
det = TopologicalAnomalyDetector(window_size=100, n_centers_by_dim=3).fit(X)
s = det.score_samples(X)
assert s.shape == (600,) and np.isfinite(s).all()
print("tdaad    ok:", s.shape)

print("ALL OK")
