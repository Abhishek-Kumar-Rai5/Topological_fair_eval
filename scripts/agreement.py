"""Agreement between the filled self re-check sheet and the master table: % agreement and Cohen's kappa.

    python scripts/agreement.py literature/recheck/filled_sheet.csv [--master literature/tda_records_master.csv]
NOTE: read the CSVs with keep_default_na=False, or pandas turns the score "NA" into a missing value.
Columns compared: R1..R13 (Y / P / N / U / NA), OUT, ISO. Free text in OUT / ISO of the master ("Tied (SMD/MSL core); ...",
"U (paper not read)") is cut to its first word, as literature/make_figs.py does for ISO; the same cut is applied to your sheet.
Reported: per column, per paper, and pooled over the R items (one pair per paper x item), over OUT and over ISO.
Kappa is computed pooled over the cells of a group (a column with 7 papers is too small for a stable kappa);
if the pooled labels are all identical the kappa is undefined and is reported as NaN.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

R_COLS = [f"R{i}" for i in range(1, 14)]


def normalise(x):
    """'Tied (SMD/MSL core); Helped ...' -> 'Tied'; ' y ' -> 'Y'; blank -> ''."""
    s = str(x).strip() if isinstance(x, str) else ""
    first = s.split()[0] if s else ""
    return first.rstrip(";,").upper() if first.upper() in {"Y", "P", "N", "U", "NA"} else first.rstrip(";,")


def agreement(sheet, master):
    """Return (per_column, per_paper, pooled) DataFrames. Papers are matched on `id`; the master is read-only."""
    ids = list(sheet["id"])
    a = sheet.set_index("id").loc[ids]
    b = master.set_index("id").loc[ids]
    cols = R_COLS + ["OUT", "ISO"]
    mine = a[cols].apply(lambda c: c.map(normalise))
    ref = b[cols].apply(lambda c: c.map(normalise))
    if (mine == "").any().any():
        raise ValueError("the sheet has empty cells: " + str(list(mine.columns[(mine == "").any()])))
    match = mine == ref
    per_col = pd.DataFrame({"agree": match.mean(), "n": len(ids)})
    per_paper = pd.DataFrame({"agree_R": match[R_COLS].mean(axis=1), "OUT": match["OUT"], "ISO": match["ISO"]})

    def pooled(name, cs):
        x, y = mine[cs].to_numpy().ravel(), ref[cs].to_numpy().ravel()
        k = cohen_kappa_score(x, y) if len(set(x) | set(y)) > 1 else np.nan
        return {"group": name, "cells": len(x), "percent_agreement": 100 * float((x == y).mean()), "kappa": float(k)}

    return per_col, per_paper, pd.DataFrame([pooled("R1-R13", R_COLS), pooled("OUT", ["OUT"]), pooled("ISO", ["ISO"]),
                                             pooled("all", R_COLS + ["OUT", "ISO"])])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--master", default="literature/tda_records_master.csv")
    args = ap.parse_args()
    pc, pp, po = agreement(pd.read_csv(args.sheet, dtype=str, keep_default_na=False), pd.read_csv(args.master, dtype=str, keep_default_na=False))
    pd.set_option("display.width", 200)
    print(pc.round(3).T.to_string(), "\n\n", pp.round(3).to_string(), "\n\n", po.round(3).to_string(index=False), sep="")
