"""E5: synthetic anomalies injected into SMD normal test segments (prereg §3, §7).

Types: S1 period change, S2 phase shift, S3 correlation break (shape);
       V1 level shift, V2 spikes, V3 variance scaling (volume).
Each injection is 60 samples long and touches ONE channel (the partner of a coupled pair is left alone).

Three steps, so the rules are easy to check:
    1. channel_profile(train)            eligible channels, from TRAINING data only
    2. plan_sites(labels, profile, ...)  where to inject, from the test LABELS and length only
    3. apply_plan(test, plan, magnitude) the modified test series (+ a mask of injected points)
Sites and channels are fixed per (machine, type), so the three magnitudes of a type are paired.
Training and validation data are never passed to steps 2-3, so they cannot be touched.

Choices the prereg leaves open (listed in docs/implementation_notes.md):
    - "dominant lag" = highest local maximum of the training autocorrelation over lags 2..30
      (a monotone, trend-like ACF has no peak, so it is not "periodic")
    - S1 factor f: the period is DIVIDED by f (the segment is read f times faster)
    - S2 shifts the periodic channel of the pair (S2 needs a period); shift = fraction * its lag
    - S3 replaces a block of round(q * 60) samples with the same block of a random label-free
      60-sample segment of the same channel (taken from the unmodified test series)
    - V1-V3 pick any channel with training std >= 0.01 (sigma = that std); near-constant channels would
      get an invisible injection. Same floor as F3
    - V3 scales the VARIANCE by f (deviations from the segment mean times sqrt(f))
    - spacing: at least 120 samples between the end of one injection and the start of the next
"""
import zlib
from dataclasses import dataclass, field

import numpy as np

TYPES = ["S1", "S2", "S3", "V1", "V2", "V3"]
KIND = {"S1": "shape", "S2": "shape", "S3": "shape", "V1": "volume", "V2": "volume", "V3": "volume"}
LENGTH = 60
MAX_LAG = 30                      # same lag range as F2's autocorrelation feature
MAX_PERIOD_FACTOR = 3             # largest S1 factor in prereg §3 (used for the padding)
STD_FLOOR = 0.01                  # V1-V3 need a channel with training std >= this (same floor as F3)


# ---------- 1. eligibility (training data only) ----------

@dataclass
class Profile:
    n_channels: int
    std: np.ndarray                    # training std per channel
    periodic: dict = field(default_factory=dict)   # channel -> dominant lag
    pairs: list = field(default_factory=list)      # (i, j) with |r| > threshold
    nonconstant: list = field(default_factory=list)
    volume: list = field(default_factory=list)     # channels V1-V3 may use: training std >= STD_FLOOR


def _acf(x, max_lag):
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return None
    return np.array([float((x[:-l] * x[l:]).sum()) / denom if l else 1.0 for l in range(max_lag + 2)])


def dominant_lag(x, acf_min=0.5, max_lag=MAX_LAG):
    """Lag (2..max_lag) of the highest ACF peak, or None if no peak above acf_min."""
    acf = _acf(np.asarray(x, float), max_lag)
    if acf is None:
        return None
    best, best_lag = acf_min, None
    for lag in range(2, max_lag + 1):
        is_peak = acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1]
        if is_peak and acf[lag] > best:
            best, best_lag = acf[lag], lag
    return best_lag


def channel_profile(train, acf_min=0.5, corr_min=0.7):
    """Eligible channels and pairs, from training data only (prereg §3)."""
    train = np.asarray(train, float)
    d = train.shape[1]
    const = np.ptp(train, axis=0) == 0            # exact "constant", as in the F2 notes
    prof = Profile(d, train.std(axis=0))
    prof.nonconstant = [c for c in range(d) if not const[c]]
    prof.volume = [c for c in range(d) if prof.std[c] >= STD_FLOOR]
    for c in prof.nonconstant:
        lag = dominant_lag(train[:, c], acf_min)
        if lag is not None:
            prof.periodic[c] = lag
    ok = prof.nonconstant
    if len(ok) > 1:
        r = np.corrcoef(train[:, ok], rowvar=False)
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                if abs(r[a, b]) > corr_min:
                    prof.pairs.append((ok[a], ok[b]))
    return prof


def eligible(itype, prof):
    """Candidate 'units' for one type: channels, or (target, partner) pairs."""
    if itype == "S1":
        return list(prof.periodic)
    if itype == "S2":                    # shift the periodic channel of a pair
        out = []
        for a, b in prof.pairs:
            out += [(c, o) for c, o in ((a, b), (b, a)) if c in prof.periodic]
        return out
    if itype == "S3":
        return [(a, b) for a, b in prof.pairs] + [(b, a) for a, b in prof.pairs]
    return list(prof.volume)             # V1-V3


# ---------- 2. where to inject (labels and length only) ----------

@dataclass
class Plan:
    itype: str
    starts: np.ndarray                  # injection start indices
    units: list                         # channel, or (target, partner), per site
    draws: list                         # per-site random numbers (fixed across magnitudes)
    n_requested: int
    reason: str = ""                    # why fewer sites than requested (empty = all placed)

    @property
    def n_placed(self):
        return len(self.starts)


def _pad(itype):
    """(samples needed before, after) the 60-sample site, so source reads stay inside clean data."""
    if itype == "S1":
        return 0, int(np.ceil((LENGTH - 1) * MAX_PERIOD_FACTOR)) + 2 - LENGTH
    if itype == "S2":
        return int(np.ceil(0.5 * MAX_LAG)) + 1, 0
    return 0, 0


def machine_seed(machine, itype, base=0):
    """Deterministic seed from (machine, type): same sites on every run and every magnitude."""
    return int(np.random.SeedSequence([zlib.crc32(machine.encode()), zlib.crc32(itype.encode()), base])
               .generate_state(1)[0])


def _clean_mask(labels, size_before, size_after, n_total, length):
    """ok[s] = True if [s - before, s + length + after) lies in the series and has no labelled point."""
    cum = np.concatenate([[0], np.cumsum(np.asarray(labels).astype(bool))])
    s = np.arange(n_total)
    lo, hi = s - size_before, s + length + size_after
    inside = (lo >= 0) & (hi <= n_total)
    lo_c, hi_c = np.clip(lo, 0, n_total), np.clip(hi, 0, n_total)
    return inside & (cum[hi_c] - cum[lo_c] == 0)


def plan_sites(test_labels, prof, itype, seed, n=10, min_gap=120):
    """Choose up to n sites in normal test segments. Uses only the point labels and their length."""
    rng = np.random.default_rng(seed)
    units_all = eligible(itype, prof)
    plan = Plan(itype, np.array([], int), [], [], n)
    if not units_all:
        plan.reason = "no eligible channel"
        return plan
    n_total = len(test_labels)
    before, after = _pad(itype)
    ok = _clean_mask(test_labels, before, after, n_total, LENGTH)
    cand = rng.permutation(np.flatnonzero(ok))
    chosen = []
    for s in cand:
        if all(abs(int(s) - c) >= LENGTH + min_gap for c in chosen):
            chosen.append(int(s))
            if len(chosen) == n:
                break
    # S3 source segment: random label-free 60-sample block, >= 60 away from the site
    src_ok = np.flatnonzero(_clean_mask(test_labels, 0, 0, n_total, LENGTH))
    for s in chosen:
        unit = units_all[int(rng.integers(len(units_all)))]
        d = {"u": rng.random(), "spikes": np.sort(rng.choice(LENGTH, 3, replace=False))}
        far = src_ok[np.abs(src_ok - s) >= LENGTH]
        d["src"] = int(far[rng.integers(len(far))]) if len(far) else None
        plan.starts = np.append(plan.starts, s)
        plan.units.append(unit)
        plan.draws.append(d)
    if len(chosen) < n:
        plan.reason = f"only {len(chosen)} of {n} sites fit"
    return plan


# ---------- 3. apply ----------

def _read(x, pos):
    """Linear interpolation of series x at (fractional) positions pos."""
    return np.interp(pos, np.arange(len(x)), x)


def _inject_one(x, s, itype, mag, unit, d, prof, original):
    """Modify column x (a copy of one channel) in place at [s, s+60)."""
    seg = slice(s, s + LENGTH)
    i = np.arange(LENGTH)
    if itype == "S1":                      # read the channel f times faster: period / f
        x[seg] = _read(original, s + mag * i)
    elif itype == "S2":                    # delay by mag * period samples
        x[seg] = _read(original, s + i - mag * prof.periodic[unit[0]])
    elif itype == "S3":                    # block of a random same-channel segment
        n_rep = int(round(mag * LENGTH))
        b = int(d["u"] * (LENGTH - n_rep + 1))
        x[s + b:s + b + n_rep] = original[d["src"] + b:d["src"] + b + n_rep]
    elif itype == "V1":                    # level shift of mag training sigmas
        x[seg] = original[seg] + mag * prof.std[unit]
    elif itype == "V2":                    # 3 spikes of mag sigmas
        x[s + d["spikes"]] = original[s + d["spikes"]] + mag * prof.std[unit]
    elif itype == "V3":                    # variance x mag, mean of the segment kept
        m = original[seg].mean()
        x[seg] = m + np.sqrt(mag) * (original[seg] - m)
    else:
        raise ValueError(itype)


def apply_plan(test, plan, magnitude, prof):
    """Return (modified test, injected mask). The input array is not changed.

    mask[t] = True for the 60 samples of every injection (also for samples a spike left unchanged).
    """
    test = np.asarray(test, float)
    out = test.copy()
    mask = np.zeros(len(test), bool)
    for s, unit, d in zip(plan.starts, plan.units, plan.draws):
        col = unit[0] if isinstance(unit, tuple) else unit
        if plan.itype == "S3" and d["src"] is None:
            continue
        _inject_one(out[:, col], int(s), plan.itype, magnitude, unit, d, prof, test[:, col])
        mask[s:s + LENGTH] = True
    return out, mask


def injection_window_labels(real_labels, injected, starts, size=60, tail=10):
    """Window labels for an injected test series: 1 / 0 / -1 (prereg §2 rule, adapted).

    positive: an injected point in the final `tail` samples and no real labelled point in the window
    normal:   no injected and no real labelled point in the window
    excluded: everything else (including every window that touches a real labelled anomaly),
              so positives are purely injected and normals are purely normal.
    """
    real, inj = np.asarray(real_labels).astype(bool), np.asarray(injected).astype(bool)
    out = np.empty(len(starts), int)
    for k, s in enumerate(starts):
        r, j = real[s:s + size], inj[s:s + size]
        if r.any():
            out[k] = -1
        elif j[-tail:].any():
            out[k] = 1
        elif not j.any():
            out[k] = 0
        else:
            out[k] = -1
    return out
