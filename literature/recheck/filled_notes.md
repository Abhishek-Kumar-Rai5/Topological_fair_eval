# Self re-check: justifications (blind, scored from full text 2026-11-21)

Scored by Claude from each paper's link only; no master table, dossier or code-verification file was opened before saving `filled_sheet.csv`.
Text was extracted with `pdftotext`. Section, table and figure numbers refer to the version at the link.

**Not accessible, so every item is scored U:**
- **D04** (doi.org/10.1145/3627673.3679614): the DOI redirects to dl.acm.org, which returned HTTP 403 (a Cloudflare challenge) for both the landing page and `/doi/pdf/`.
- **B03** (openreview.net/pdf?id=BXGqPm6nKgP): curl got HTTP 403, a second fetch got a "Verifying your browser" challenge page, and the OpenReview API returned JSON, not the PDF.

## C01: Davies 2022, TDA for host-based logs (arXiv 2204.12919v1)
- R1 Y: counts baseline on the same runs, with the same 10-fold CV and the same default RF (§3, App. B.1, B.5, Table 6).
- R2 Y: PH persistence images are swapped for the counts vector (and Laplacian spectra) under the same RF (§4, Tables 1–2).
- R3 N: random 10-fold CV over runs. Dataset 2 obfuscated copies of the same run fall across folds, and the author admits train/test overlap (§4, App. B.5).
- R4 Y: supervised RF with default predict. No threshold is tuned on test data (App. B.5).
- R5 N: only accuracy, precision, recall and F1 on balanced classes (Tables 1–2).
- R6 N: ± is across CV folds only. The two datasets are reported separately, with no CI across units.
- R7 N: no timings. There is only a qualitative remark that more logs cost more compute (§4).
- R8 N: none.
- R9 N: none.
- R10 P: acknowledges synthetic balanced data and Dataset 2 overlap (§4, §6, App. B.5), but makes no correction.
- R11 P: the MDI feature-importance plots (§5, Fig. 3) back the "born early, persist" reading, but the author says the meaning "requires further investigation".
- R12 N: no code link.
- R13 N: "two different private datasets" (§3.1).
- OUT Lost: PH vs counts F1 is 66.6 vs 72.5 and 65.0 vs 73.2 on Dataset 1 (Table 1). On Dataset 2 it is 74.8 vs 73.6 and 78.1 vs 76.9, within about ±1 (Table 2). So PH loses on one dataset and ties on the other. The best non-topological arm (graph Laplacian or Counts+GL) beats PH everywhere.
- ISO Replacement: separate feature sets go into the same classifier.

## C02: Tidjon & Khomh 2022, malware TDA (arXiv 2211.01535v2)
- R1 P: PCA, UMAP and t-SNE features go through the same classifiers on the same datasets (§IV-C). The train/test split is never described, and Optuna tuning is not stated per arm.
- R2 P: persistence-diagram features are swapped for PCA, UMAP and t-SNE (§IV-C, Figs. 9–12). But the PD encoding is obscure ("homology group features as classes … relabeled data cloud points"), so it is unclear what is actually swapped.
- R3 N: min-max scaling and label encoding are applied to the full dataset before analysis (§IV-B, §IV-C), and no split is described.
- R4 U: the evaluation protocol and threshold are not reported.
- R5 N: detection rate and FPR only (§IV-D1). Precision is "in replication package".
- R6 N: no variance or CI. Runs were repeated "2 to 3 times" for correctness only (§VIII).
- R7 Y: CPU time, wall time and memory for extraction, training and inference (Tables I–X).
- R8 Y: additive Gaussian noise with α ∈ {0.001, 0.01, 0.1, 1} (§IV-A3, RQ3, §V-C).
- R9 N: none. "Bypass detectors" appears in the motivation only.
- R10 N: CIC datasets are chosen for "reputation", and no flaws are discussed.
- R11 N: claims that Mapper "identified a new malware class Zero-day" and shows "hidden relationships" (§IV-B4, §VI) rest on visual inspection only.
- R12 Y: replication package on GitHub (ref [45]).
- R13 Y: CIC-MalMem-2022 and CCCS-CIC-AndMal-2020 are public (footnotes 1–2).
- OUT Mixed: t-SNE beats PD on detection rate (99–100% vs 80–100% on MalMem, §V-B2). PD has lower FPR than PCA and is more robust at α=1 than t-SNE and UMAP for RF and DT (§V-B1, §V-C). XGBoost and LightGBM do worse with PD.
- ISO Replacement: alternative feature extractors are compared under the same classifiers.

## H04: Chazal, Levrard & Royer 2024, TADA (JMLR 25)
- R1 Y: Spectral-IF and Spectral-RF use spectral features of the same correlation graphs with the same IF/RF learners, window and stride as Atol-IF and Atol-RF (§4 intro, §4.1).
- R2 Y: topological features are replaced by spectral features under the same learner (§4 intro: "to investigate the differences … spectral analysis").
- R3 P: on Topological Wheels, models are trained on one dataset and tested on the other nine (§4.1). The GutenTAG and Exathlon protocol is not detailed ("same calibration", §4.2–4.3).
- R4 NA: only threshold-free metrics (ROC AUC, Range PR AUC). The 1−α quantile on normal data is proposed but not used in results (§2.3, §3.2).
- R5 Y: Range PR AUC is the main metric (§4 intro).
- R6 Y: results across 10, 1174 and 15 independent datasets with critical-difference tests (Figs. 8, 9, 12; Tables 1–3).
- R7 Y: median and IQR compute times on a laptop (Tables 1–3). There is no real-time latency claim.
- R8 N: no noise, drift or missing-data tests. Robustness to a corrupted base sample is argued theoretically (MCD, §2.3) only.
- R9 N: none.
- R10 Y: the GutenTAG generation artifact is acknowledged and corrected data is used (§4.2, footnote 1).
- R11 P: center-targeted scores on one synthetic Wheels set match the injected H1 change (§4.4, Fig. 14). The real-data interpretations (e.g. H0 shift means sensor breakdown) are speculative.
- R12 Y: Confiance.ai catalog record (§4 intro).
- R13 Y: Wheels in gudhi-data, plus public GutenTAG and Exathlon.
- OUT Mixed: TADA wins on Wheels, where non-topological methods fail (§4.1). On GutenTAG it ranks 4th and SubKNN is best (§4.2). On Exathlon it is 3rd by CD ranking but has the most rank-1 results (§4.3).
- ISO Replacement: the Spectral-IF/RF vs Atol-IF/RF swap.

## I03: Myers et al. 2023, zigzag persistence on OpTC (arXiv 2309.08010v1)
- R1 Y: a second autoencoder on hypergraph summary statistics, with the same hosts, windows, split and the same 2-d latent architecture (§III-B).
- R2 Y: ACC zigzag vectors are swapped for summary statistics (edges, vertices, components, diameter) (§III-B, Table III).
- R3 Y: entity-separated. The model is trained on 12 benign hosts and tested on malicious hosts 201, 402 and 660 (§III-A, Table II).
- R4 NA: no detection threshold is set. Only the reconstruction-loss distributions are compared (§IV, Table III).
- R5 N: only loss percentiles and plots. No PR-AUC or false-alarm rate.
- R6 N: one example IP and host (Figs. 7–8). Table III covers host 201 only, with no CI.
- R7 N: none.
- R8 N: none. Sensitivity analysis is future work (§V).
- R9 N: none.
- R10 P: notes that OpTC is a simulated test environment whose benign activity may differ from real traffic (§III-C).
- R11 P: the Figs. 5–6 intuition for one IP supports the "faster dynamics" claim, but the authors say interpreting H0/H1 features "is ongoing work" (§V).
- R12 N: no code link.
- R13 Y: DARPA OpTC is public, and the ground truth is on GitHub (§III-A, footnote 3).
- OUT Helped: the median malicious/benign loss ratio is 35.6× for ACC vs 7.5× for summary statistics, and ACC localises the first attack burst more sharply (§IV, Table III, Fig. 7). This is a weak result: one host, no test.
- ISO Replacement.

## N10: Jeet, Karn & Kumar 2026, TDA+LSTM on CIC-IDS2017 (arXiv 2606.31619v1)
- R1 Y: LSTM-only on the same subsample, split and training strategy (Table VIII, §V-D).
- R2 N: no cheap-feature replacement. TDA is only removed (LSTM-only), and SVM and IF are different learners on raw features.
- R3 N: a stratified random 80/20 split of overlapping windows (window 50, step 25), with min-max scaling listed before the split (§V-B, §VI-A).
- R4 Y: softmax argmax. No threshold is tuned on test data (§IV-D).
- R5 N: AUC, F1, precision and recall only (Tables I, VII, VIII).
- R6 N: 5-fold CV and McNemar on one dataset only (Tables VI–VII).
- R7 P: training time and memory are given (Table III). "Near real-time capability" is claimed (§IX-B) without latency at a stated event rate.
- R8 N: robustness is asserted from the stability theorem only (App. A-B, §IX-C). It is not tested.
- R9 N: adversarial obfuscation is mentioned in §I but not tested.
- R10 N: CIC-IDS2017 is called "a top standard" (§II-D). Known labelling and feature flaws are not acknowledged.
- R11 N: claims of interpretable "topological signatures" per attack type rest on figures with 6 and 1 test windows (Fig. 7, Table II). Betti-1 importance is 0.0 (Table V).
- R12 N: no code link.
- R13 Y: CIC-IDS2017 is public.
- OUT Tied: hybrid vs LSTM-only is F1 1.000 vs 1.000 and AUC 1.000 vs 1.000 (Table VIII). McNemar gives p=1.0 (Table VI).
- ISO Removal-only: the ablation is TDA-only, LSTM-only and Hybrid (Table VIII).

## D04: Liu et al. 2024, CIKM (doi 10.1145/3627673.3679614)
- All items U: full text not accessible (ACM 403 / Cloudflare challenge).

## B03: Collins et al. 2020, IoT fingerprinting (OpenReview BXGqPm6nKgP)
- All items U: full text not accessible (OpenReview 403 / browser-verification page).
