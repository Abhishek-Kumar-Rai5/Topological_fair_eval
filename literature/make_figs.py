"""Figures v2 from tda_records_master.csv. Run: python3 make_figs_v2.py <master.csv> <outdir>"""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
src, out = sys.argv[1], sys.argv[2]
m = pd.read_csv(src, dtype=str, keep_default_na=False)
df = m[m.inclusion_status.str.startswith("core") & ~m.inclusion_status.str.startswith("core-method") & (m.id != "D03")].copy()
RC = [f"R{k}" for k in range(1, 14)]
VALS = ["Helped", "Tied", "Mixed", "Lost", "Untested", "Claimed, unverified", "Unclear"]
def norm(o, ver):
    o = o.strip()
    if o in ("", "U"): return "Unclear"
    if o.startswith("Untested"): return "Untested"
    if ver == "abstract" or "abstract" in o.lower(): return "Claimed, unverified"
    return "Tied" if o.startswith("Tied") else o.split()[0]
df["out"] = [norm(o, v) for o, v in zip(df.OUT, df.verification)]
df["iso"] = df.ISO.str.split().str[0].replace({"U": "Unknown", "": "Unknown"})
assert set(df.out) <= set(VALS)
n, n_full = len(df), (df.verification == "full_text").sum()
order = ["S1","S2","S3","S4","S5","S6","S7","S8","I1","I2","I3","I4","I5","I6"]

# ---- Fig 1: (a) OUT by cluster, (b) OUT x ISO for full-text papers with a comparison
col = {"Helped":"#08306B","Tied":"#4292C6","Mixed":"#E69F00","Lost":"#B2182B","Untested":"#D9D9D9","Claimed, unverified":"#FFFFFF","Unclear":"#8C8C8C"}
hat = {"Helped":"","Tied":"","Mixed":"//","Lost":"xx","Untested":"","Claimed, unverified":"..","Unclear":""}
ct = pd.crosstab(df.cluster, df.out).reindex(columns=VALS, fill_value=0)
ct = ct.loc[[c for c in order if c in ct.index]]; ct["total"] = ct.sum(axis=1); ct = ct.sort_values("total")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
left = np.zeros(len(ct))
for v in VALS:
    ax.barh(ct.index, ct[v], left=left, color=col[v], edgecolor="black", lw=.6, hatch=hat[v], label=v, height=.7)
    for y, (l, w) in enumerate(zip(left, ct[v])):
        if w > 0: ax.text(l + w/2, y, str(w), ha="center", va="center", fontsize=8,
                          color="white" if v in ("Helped","Lost","Tied") else "black",
                          bbox=dict(fc=col[v], ec="none", pad=1.2) if hat[v] else None)
    left += ct[v].values
for y, t in enumerate(ct.total): ax.text(t + .15, y, f"n={t}", va="center", fontsize=8)
ax.set_xlim(0, ct.total.max() + 1.3); ax.set_xlabel("Core papers (n)")
ax.set_title("(a) Outcome vs best non-topological comparator, by cluster", fontsize=9.5)
ax.spines[["top","right"]].set_visible(False)
ax.legend(title="OUT", ncol=4, loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False, fontsize=8, title_fontsize=8)
iso_order = ["Replacement", "Removal-only", "Not-isolated"]
sub = df[(df.verification == "full_text") & df.out.isin(["Helped","Tied","Mixed","Lost"])]
ci = pd.crosstab(sub.iso, sub.out).reindex(index=iso_order, columns=["Helped","Tied","Mixed","Lost"], fill_value=0)
left = np.zeros(len(ci))
for v in ci.columns:
    bx.barh(ci.index, ci[v], left=left, color=col[v], edgecolor="black", lw=.6, hatch=hat[v], height=.6)
    for y, (l, w) in enumerate(zip(left, ci[v])):
        if w > 0: bx.text(l + w/2, y, str(w), ha="center", va="center", fontsize=8,
                          color="white" if v in ("Helped","Lost","Tied") else "black",
                          bbox=dict(fc=col[v], ec="none", pad=1.2) if hat[v] else None)
    left += ci[v].values
for y, (iso, r) in enumerate(ci.iterrows()):
    ids = ", ".join(sub[sub.iso == iso].assign(o=lambda d: d.out.map(["Helped","Tied","Mixed","Lost"].index)).sort_values(["o","id"]).id)
    bx.text(left[y] + .1, y, ids, va="center", fontsize=7)
bx.invert_yaxis(); bx.set_xlim(0, ci.sum(axis=1).max() + 4.5); bx.set_xlabel("Full-text papers with a comparison (n)")
bx.set_title("(b) Same outcomes by how topology was isolated", fontsize=9.5)
bx.spines[["top","right"]].set_visible(False)
fig.text(.01, .01, f"Core: {n} papers ({n_full} full text, {n-n_full} abstract only). Excluded: boundary rows, J06; D05 merged into D02; B10 held as candidate (inclusion unverified). "
         "D03 excluded (Mapper described, no output shown).\n'Claimed, unverified' = outcome known only from abstract or repository (D02: repo numbers, paper unread). "
         "Panel (b): only Replacement isolates topology; I03's win is weak (one host, no detection metric); C02's comparison appears invalid.", fontsize=7.2, va="bottom")
fig.subplots_adjust(left=.05, right=.98, top=.92, bottom=.27, wspace=.25)
for e in ("pdf", "png"): fig.savefig(f"{out}/fig1_evidence_map.{e}", dpi=300)

# ---- Fig 2: rigor heatmap
cl = {"R1":"Matched baseline","R2":"Replacement","R3":"Leakage-safe split","R4":"Clean threshold","R5":"Realistic metrics","R6":"Real uncertainty","R7":"Cost","R8":"Robustness","R9":"Adversary","R10":"Benchmark validity","R11":"Interpretation","R12":"Code","R13":"Data"}
df["ci"] = df.cluster.map(order.index)
d2 = df.sort_values(["ci", "id"]).reset_index(drop=True)
cc = {"Y":"#08306B","P":"#7BB6D8","N":"#D55E00","U":"#DADADA","NA":"#F7F7F7"}
tc = {"Y":"white","P":"black","N":"white","U":"black","NA":"#555555"}
fig, ax = plt.subplots(figsize=(9.5, 11.6))
for i, r in d2.iterrows():
    for j, k in enumerate(RC):
        v = r[k]; ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=cc[v], ec="white", lw=1.2))
        ax.text(j + .5, i + .5, v, ha="center", va="center", fontsize=8, color=tc[v])
for row, (lab, f) in enumerate([("% Y (all rows)", lambda s: (s == "Y").sum() / len(s)),
                                 ("% Y (applicable, excl. NA)", lambda s: (s == "Y").sum() / max((s != "NA").sum(), 1))]):
    y = n + .6 + row * 1.2
    for j, k in enumerate(RC):
        ax.add_patch(plt.Rectangle((j, y), 1, 1, fc="white", ec="black", lw=.8))
        ax.text(j + .5, y + .5, f"{100*f(d2[k]):.0f}%", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(-.15, y + .5, lab, ha="right", va="center", fontsize=8.5, fontweight="bold")
ax.set_yticks(np.arange(n) + .5)
ax.set_yticklabels([r.id  + (" †" if r.id == "D02" else "") for r in d2.itertuples()], fontsize=8)
ax.set_xticks(np.arange(13) + .5); ax.set_xticklabels([f"{k} {cl[k]}" for k in RC], rotation=50, ha="left", fontsize=8.5)
ax.xaxis.tick_top(); ax.tick_params(length=0); ax.set_xlim(0, 13); ax.set_ylim(n + 3.2, 0)
start = 0
for c in order:
    k = (d2.cluster == c).sum()
    if k == 0: continue
    if start > 0: ax.axhline(start, color="black", lw=1.2)
    ax.text(-.19, start + k/2, c, ha="center", va="center", fontsize=9, fontweight="bold", transform=ax.get_yaxis_transform())
    start += k
for s in ax.spines.values(): s.set_visible(False)
ax.legend(handles=[Patch(fc=cc[k], ec="black", label=k) for k in cc], ncol=5, loc="lower center", bbox_to_anchor=(.5, -.07),
          frameon=False, fontsize=8, title="Y yes · P partial · N no · U unknown (not reported / not readable) · NA not applicable", title_fontsize=8)
fig.text(.01, .005, f"n={n} core papers, cluster order S1–I6. † D02 scored from repository README; paper unread. "
         "Unscored boundary rows are excluded, so every NA here means 'not applicable'.", fontsize=7.2)
fig.subplots_adjust(left=.2, right=.97, top=.86, bottom=.08)
for e in ("pdf", "png"): fig.savefig(f"{out}/fig2_rigor_heatmap.{e}", dpi=300)

# ---- counts
o = []
for c, r in ct.iterrows():
    for v in VALS: o.append(("fig1a", c, v, int(r[v])))
    o.append(("fig1a", c, "TOTAL", int(r.total)))
for iso, r in ci.iterrows():
    for v in ci.columns: o.append(("fig1b", iso, v, int(r[v])))
o += [("fig1", "ALL", "full_text", int(n_full)), ("fig1", "ALL", "abstract_only", int(n - n_full))]
for k in RC:
    s = d2[k]
    for v in ["Y","P","N","U","NA"]: o.append(("fig2", f"{k} {cl[k]}", v, int((s == v).sum())))
    o.append(("fig2", f"{k} {cl[k]}", "pct_Y_all", round(100*(s == "Y").mean(), 1)))
    o.append(("fig2", f"{k} {cl[k]}", "pct_Y_applicable", round(100*(s == "Y").sum()/max((s != "NA").sum(), 1), 1)))
pd.DataFrame(o, columns=["figure","group","category","count"]).to_csv(f"{out}/figure_counts.csv", index=False)
print(ct.total.sum(), n_full, "\n", ci)