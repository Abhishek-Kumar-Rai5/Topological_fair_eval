"""Blank self re-check sheet (prereg step 4): the 7 papers with title and link, and EMPTY score columns.

    python scripts/make_blank_sheet.py
Writes literature/recheck/blank_sheet.csv. No existing score, outcome, group, status or note is copied from the master
table, so the re-score is blind. Definitions of the columns: literature/recheck/definitions.md.
"""
from pathlib import Path

import pandas as pd

IDS = ["C01", "C02", "H04", "I03", "D04", "N10", "B03"]          # as listed by Anirudh, in this order
SCORES = [f"R{i}" for i in range(1, 14)] + ["OUT", "ISO"]


def blank_sheet(master_path="literature/tda_records_master.csv", ids=IDS):
    m = pd.read_csv(master_path).set_index("id")
    rows = []
    for i in ids:
        r = m.loc[i]
        link = next((str(r[c]) for c in ("url", "doi", "arxiv") if isinstance(r[c], str) and r[c].strip()), "")
        if link and not link.startswith("http") and r["doi"] == link:
            link = "https://doi.org/" + link
        rows.append({"id": i, "title": r["title"], "authors": r["authors"], "year": r["year"], "link": link,
                     **{c: "" for c in SCORES}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path("literature/recheck/blank_sheet.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = blank_sheet()
    df.to_csv(out, index=False)
    print("wrote", out, f"({len(df)} papers)")
