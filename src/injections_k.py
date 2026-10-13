"""E5b (exploratory): the E5 injections, but into K = 5 channels at once, the same K for every type.

Same six types, magnitudes, 60-sample length, spacing and window labelling as E5 (src/injections.py).
What changes: every injection site touches K channels, chosen from the eligible ones (training data only):
    S1  period change          K periodic channels, same factor f
    S2  phase shift            K periodic channels that each have a coupled partner; same fraction of each
                               channel's own period; the partners are untouched
    S3  correlation break      K coupled pairs; the target channel's block is replaced by the same block of
                               one random label-free segment of that channel (same source time for all K)
    V1  level shift            K channels (training std >= 0.01), each + mag * its own training sigma
    V2  spikes                 K channels, the same 3 random time points, each + mag * its sigma
    V3  variance x mag         K channels, each around its own segment mean
For S2 and S3 the K pairs must be DISJOINT (no channel is a target and also a partner), otherwise the
coupling inside the group would survive. A machine with fewer than K usable channels (or disjoint pairs) for a
type gets no sites for that type; the placement file says why (nothing is silent).
Sites, channels and random draws depend on (machine, type) only, so the three magnitudes are paired.
"""
import numpy as np

from src.injections import (LENGTH, Plan, _clean_mask, _inject_one, _pad, eligible)

K = 5
TRIES = 20                              # attempts to find K disjoint pairs before giving up


def choose_units(itype, prof, k, rng):
    """k units for one site (channels, or (target, partner) pairs), or None if they cannot be found."""
    cands = eligible(itype, prof)
    if itype in ("S1", "V1", "V2", "V3"):          # channels
        if len(cands) < k:
            return None
        return [cands[i] for i in rng.choice(len(cands), k, replace=False)]
    for _ in range(TRIES):                          # S2, S3: k disjoint pairs, greedy in random order
        targets, partners, out = set(), set(), []
        for i in rng.permutation(len(cands)):
            t, p = cands[i]
            if t in targets or t in partners or p in targets:
                continue
            out.append((t, p)); targets.add(t); partners.add(p)
            if len(out) == k:
                return out
    return None


def n_eligible(itype, prof):
    """How many channels could be touched (distinct target channels for S2 and S3)."""
    c = eligible(itype, prof)
    return len(c) if itype in ("S1", "V1", "V2", "V3") else len({t for t, _ in c})


def plan_sites_k(test_labels, prof, itype, seed, k=K, n=10, min_gap=120):
    """Up to n sites in normal test segments, each with k units. Uses only the labels and their length."""
    rng = np.random.default_rng(seed)
    plan = Plan(itype, np.array([], int), [], [], n)
    if choose_units(itype, prof, k, np.random.default_rng(seed + 1)) is None:
        plan.reason = f"fewer than {k} usable channels ({n_eligible(itype, prof)} eligible)"
        return plan
    n_total = len(test_labels)
    before, after = _pad(itype)
    ok = _clean_mask(test_labels, before, after, n_total, LENGTH)
    chosen = []
    for s in rng.permutation(np.flatnonzero(ok)):
        if all(abs(int(s) - c) >= LENGTH + min_gap for c in chosen):
            chosen.append(int(s))
            if len(chosen) == n:
                break
    src_ok = np.flatnonzero(_clean_mask(test_labels, 0, 0, n_total, LENGTH))
    for s in chosen:
        units = choose_units(itype, prof, k, rng)
        d = {"u": rng.random(), "spikes": np.sort(rng.choice(LENGTH, 3, replace=False))}
        far = src_ok[np.abs(src_ok - s) >= LENGTH]
        d["src"] = int(far[rng.integers(len(far))]) if len(far) else None
        plan.starts = np.append(plan.starts, s)
        plan.units.append(units)
        plan.draws.append(d)
    if len(chosen) < n:
        plan.reason = f"only {len(chosen)} of {n} sites fit"
    return plan


def apply_plan_k(test, plan, magnitude, prof):
    """Return (modified test, injected mask) with every site touching all of its k channels."""
    test = np.asarray(test, float)
    out = test.copy()
    mask = np.zeros(len(test), bool)
    for s, units, d in zip(plan.starts, plan.units, plan.draws):
        if plan.itype == "S3" and d["src"] is None:
            continue
        for unit in units:
            col = unit[0] if isinstance(unit, tuple) else unit
            _inject_one(out[:, col], int(s), plan.itype, magnitude, unit, d, prof, test[:, col])
        mask[s:s + LENGTH] = True
    return out, mask
