"""CTU-13: flows -> per-host windows, labels and destination graphs (prereg §2, §4; notes 49-57).

    read_scenario(folder)   two passes over the .binetflow(.xz): hosts, then the flows of those hosts
    host_windows(...)       10 s bins, 60 bins per window, step 30; the 6 channels and the label per window
    window_graph(...)       bipartite destination graph (destination IPs - destination ports) of one window

Only reads the data. Nothing is written under data/.
"""
import glob
import html
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BIN_SECONDS, WINDOW_BINS, STEP_BINS = 10, 60, 30
CHANNELS = ["flows", "packets", "bytes", "src_bytes", "unique_dst_ip", "unique_dst_port"]
MAX_EDGES = 2000                                   # cap on graph edges per window (note 56)
FIRST_CAPTURE = 42                                 # capture 42 = scenario 1
POSITIVE, NORMAL = 1, 0
CHUNK = 500_000


def label_class(label):
    """'flow=From-Botnet-V42-TCP-CC6' -> 'From-Botnet';  'flow=Background-TCP' -> 'Background'."""
    tokens = str(label).removeprefix("flow=").split("-")
    return "-".join(tokens[:2]) if tokens[0] in ("From", "To") else tokens[0]


def infected_ips(folder):
    """IPs the README lists as 'IP: x (Label: Botnet)'."""
    files = glob.glob(str(Path(folder) / "README*"))
    text = html.unescape(re.sub(r"<[^>]+>", " ", Path(files[0]).read_text(errors="replace")))
    return sorted(set(re.findall(r"IP:\s*(\d+\.\d+\.\d+\.\d+)\s*\(Label:\s*Botnet\)", text)))


def scenario_folder(scenario, root="data/CTU-13"):
    return Path(root) / str(FIRST_CAPTURE + scenario - 1)


def flow_file(folder):
    files = sorted(Path(folder).glob("*.binetflow*"))
    if not files:
        raise FileNotFoundError(f"no .binetflow in {folder}")
    return files[0]


@dataclass
class Scenario:
    flows: pd.DataFrame        # flows sent by the listed hosts: host, bin, dst, dport, pkts, bytes, src_bytes, cls
    infected: list
    normal: list
    n_bins: int                # bins in the whole file (bin 0 = first flow)
    t0: pd.Timestamp


def read_scenario(folder, infected=None):
    """Pass 1: capture span and the normal hosts. Pass 2: the flows of infected + normal hosts."""
    folder = Path(folder)
    path = flow_file(folder)
    infected = list(infected if infected is not None else infected_ips(folder))
    t_min = t_max = None
    normal = set()
    for ch in pd.read_csv(path, usecols=["StartTime", "SrcAddr", "Label"], chunksize=CHUNK, dtype=str):
        lo, hi = ch["StartTime"].min(), ch["StartTime"].max()     # fixed-width strings sort in time order
        t_min = lo if t_min is None or lo < t_min else t_min
        t_max = hi if t_max is None or hi > t_max else t_max
        normal.update(ch.loc[ch["Label"].fillna("").str.startswith("flow=From-Normal"), "SrcAddr"].unique())
    normal = sorted(normal - set(infected))
    t0 = pd.to_datetime(t_min, format="%Y/%m/%d %H:%M:%S.%f")
    n_bins = int((pd.to_datetime(t_max, format="%Y/%m/%d %H:%M:%S.%f") - t0).total_seconds() // BIN_SECONDS) + 1
    hosts = set(infected) | set(normal)
    parts = []
    use = ["StartTime", "SrcAddr", "DstAddr", "Dport", "TotPkts", "TotBytes", "SrcBytes", "Label"]
    for ch in pd.read_csv(path, usecols=use, chunksize=CHUNK, dtype={"Dport": str, "DstAddr": str, "SrcAddr": str}):
        ch = ch[ch["SrcAddr"].isin(hosts)]
        if len(ch):
            parts.append(ch)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use)
    t = pd.to_datetime(df["StartTime"], format="%Y/%m/%d %H:%M:%S.%f")
    flows = pd.DataFrame({
        "host": df["SrcAddr"].to_numpy(),
        "bin": ((t - t0).dt.total_seconds() // BIN_SECONDS).astype(int).to_numpy(),
        "dst": df["DstAddr"].fillna("?").to_numpy(), "dport": df["Dport"].fillna("?").to_numpy(),
        "pkts": df["TotPkts"].to_numpy(float), "bytes": df["TotBytes"].to_numpy(float),
        "src_bytes": df["SrcBytes"].to_numpy(float), "cls": df["Label"].fillna("").map(label_class).to_numpy()})
    return Scenario(flows.sort_values(["host", "bin"], kind="stable").reset_index(drop=True), infected, normal, n_bins, t0)


# ---------------------------------------------------------------------------------------- windows
def window_range(host_flows, n_bins, size=WINDOW_BINS, step=STEP_BINS):
    """Window indices j (bins step*j .. step*j+size-1) from the host's first to its last flow; full windows only."""
    if len(host_flows) == 0:
        return np.array([], int)
    b_min, b_max = int(host_flows["bin"].min()), int(host_flows["bin"].max())
    j_first = max(0, -(-(b_min - size + 1) // step))            # first j with step*j + size - 1 >= b_min
    j_last = min(b_max // step, (n_bins - size) // step)         # full windows only
    return np.arange(j_first, j_last + 1) if j_last >= j_first else np.array([], int)


def window_flows(host_flows, j, size=WINDOW_BINS, step=STEP_BINS):
    b = host_flows["bin"].to_numpy()
    lo, hi = np.searchsorted(b, [step * j, step * j + size])    # host_flows is sorted by bin
    return host_flows.iloc[lo:hi]


def channel_matrix(wf, j, size=WINDOW_BINS, step=STEP_BINS):
    """(size, 6) array of the per-bin channels of one window (note 53)."""
    out = np.zeros((size, len(CHANNELS)))
    if len(wf) == 0:
        return out
    rel = wf["bin"].to_numpy() - step * j
    g = wf.assign(rel=rel).groupby("rel")
    agg = g.agg(flows=("host", "size"), packets=("pkts", "sum"), bytes=("bytes", "sum"),
                src_bytes=("src_bytes", "sum"), uip=("dst", "nunique"), uport=("dport", "nunique"))
    out[agg.index.to_numpy()] = agg[["flows", "packets", "bytes", "src_bytes", "uip", "uport"]].to_numpy()
    return out


def window_label(wf, host_is_infected):
    """§2 label of one window: 1, 0 or -1 (excluded). notes 52."""
    n_bot = int((wf["cls"] == "From-Botnet").sum())
    n_normal = int((wf["cls"] == "From-Normal").sum())
    if host_is_infected:
        return POSITIVE if n_bot >= 1 else -1
    return NORMAL if (n_normal >= 1 and n_bot == 0) else -1


# ---------------------------------------------------------------------------------------- graphs
def window_graph(wf, max_edges=MAX_EDGES):
    """Bipartite destination graph of one window.

    Returns (edges, capped): edges = array (E, 2) of node-name pairs (dst IP, 'port:<p>') sorted by name,
    with at most max_edges of them (the most flows first, ties by first appearance); capped = True if cut.
    """
    if len(wf) == 0:
        return np.empty((0, 2), dtype=object), False
    pairs = wf.groupby(["dst", "dport"], sort=False).size()          # first-appearance order, counts
    order = np.argsort(-pairs.to_numpy(), kind="stable")
    capped = len(order) > max_edges
    keep = pairs.index[order[:max_edges]]
    edges = np.array([(ip, f"port:{p}") for ip, p in keep], dtype=object)
    return edges, capped
