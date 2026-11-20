# Report (LaTeX)

Build from this folder:

    latexmk -pdf main.tex        # clean up afterwards: latexmk -c

Layout
- `main.tex`: preamble, cover page, abstract, and section order.
- `sections/01..08_*.tex`: one file per report section. Sections 1-5 are drafted; 6-8 are skeletons.
- `sections/A_appendix.tex`: decision log, table of the 32 core papers, deviations log, AI-use note.
- `references.bib`: hand-written references (datasets, methods, repositories).
- `records.bib` and `tables/core_papers.tex`: generated from the literature CSV. Regenerate both from the repo root:

      python report/make_tables.py literature/tda_records_master.csv

- Figures 1 and 2 are read from `../literature/figures/` (made by `literature/make_figs.py`).

Conventions
- `\fillin{...}` prints a red TODO; `\pending` marks a result cell. Before submitting, both must be gone:
  `grep -n "fillin\|pending" sections/*.tex main.tex`
- `\rec{C01}` prints a record ID together with its citation.
- Every result number in Section 6 must come from a file in `results/` with its `.meta.json`.

Overleaf: upload this folder, plus a `figures/` subfolder holding `fig1_evidence_map.pdf` and `fig2_rigor_heatmap.pdf`.
