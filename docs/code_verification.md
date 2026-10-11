# Code-level verification: TAME, TopoGDN, MSF-IDS, Fluent Bit tda

- **Method:** the public repositories were read at a fixed commit. No code was installed or run.
- **Date:** 13 Nov 2026.
- **Scope of claims:** every claim below describes what the *released code* does. Each one links to the exact file and lines, so anyone can check it. None of these is a claim about what the authors ran for their paper.

---

## 1. TAME (D02)

**Repository:** github.com/SPIRE-Lab-2025/TAME
**Commit:** `e77c0d93434bdd6b20db9db60e524a30a20a35b5` (2026-07-18)
**Status:** ✅ Confirmed. Precise wording is below.

| # | What the code does | Where |
|---|---|---|
| 1 | All training labels are set to 0 for SMD, PSM and SMAP/MSL. | `utils/data_loaders.py` L52, L176, L224, L268 |
| 2 | On the official splits, the validation set is the last 18% of the training pool, so it contains only normal windows. | `utils/experiment_core.py` L84–L87 |
| 3 | If validation lacks both classes, the threshold falls back to the `1 − expected_rate` quantile of the *training* scores. | `utils/experiment_core.py` L105–L112 |
| 4 | `expected_rate` is the anomaly rate of the *test* labels. | `utils/experiment_core.py` L216 |
| 5 | The baselines are given the same rule. | `utils/baseline_core.py` L50 |
| 6 | "TAME (best)" is the maximum mean test AUROC across detectors and settings. Each baseline is a single model. | `consolidate.py` L66–L71 |
| 7 | The notebooks pick the plotted detector by test AUROC and set the plotted threshold from the test anomaly rate. | `notebooks/SMD.ipynb`, `PSM.ipynb`, `CATS.ipynb` |
| 8 | By contrast, the README says thresholds use "validation-only threshold selection (never test labels)". | `README.md` L40; docstring at `experiment_core.py` L106 |

**Impact**
- **Threshold-dependent metrics (F1, precision, recall):** on SMD, PSM and SMAP/MSL, these use the test anomaly rate. The same applies to the baselines, so this point by itself does not favour TAME.
- **AUROC and AUPRC:** these do not depend on the threshold.
- **Headline comparison:** the TAME number is a best-of-detectors pick made on the test set, while each baseline is a single model. This is a selection effect that favours TAME.
- The CATS dataset uses a 70/15/15 split, so its validation set may contain anomalies and the fallback may not apply there.

**Suggested wording:**
> "In the released code (commit e77c0d9), when the validation split contains no anomalies, as on the official SMD/PSM/SMAP-MSL splits, the decision threshold is set from the test-set anomaly rate (experiment_core.py L105–L112, L216); the reported 'TAME (best)' AUROC is the maximum over detectors selected on test AUROC (consolidate.py L66–L71)."

---

## 2. TopoGDN (D04)

**Repository:** github.com/ljj-cyber/TopoGDN
**Commit:** `2b8df3b2a4092a49e40b69163e0ad0d21584bab1` (2024-12-26)
**Status:** ✅ Confirmed. Fairness caveat applies.

| # | What the code does | Where |
|---|---|---|
| 1 | The default reporting mode is `-report best`. | `main.py` L349 |
| 2 | In that mode the "best" info is reported. | `main.py` L175–L185 |
| 3 | The "best" info scores F1 at 400 thresholds against the *test labels* and reports the maximum. | `evaluate.py` L129–L143 |
| 4 | Error scores are normalised using the median and IQR of *test-set* prediction errors. These come from test sensor values, not labels, so the normalisation is transductive. | `evaluate.py` L48–L60 |
| 5 | **Inherited:** the original GDN code has the same default. | d-ailin/GDN @ `9853899da860682669a134e4af315d036aab4eca`, `main.py` L216 |

**Impact:** the F1 values that the default script reports are best-case figures, chosen using test labels. Any F1 margin over TranAD therefore depends on how TranAD was thresholded. This practice is inherited from GDN and is common in this benchmark line. Present it as a field-level issue, not one specific to TopoGDN.

**Still to check:** the earlier report says the paper's Sec. 3.6 describes a validation-based threshold. I have not re-read the paper's text. We must confirm this before claiming that the paper and the code disagree.

**Suggested wording:**
> "TopoGDN's released evaluation script, inherited from GDN, by default reports the maximum F1 over thresholds evaluated on test labels (main.py L349; evaluate.py L129–L143)."

---

## 3. MSF-IDS (N03)

**Repository:** github.com/qcydm/MSF-IDS
**Commit:** `bb8ccf51c4a5d7cda66861bba30e62985cc3379d` (2024-09-24)
**Status:** ✅ Confirmed.

| # | What the code does | Where |
|---|---|---|
| 1 | For the NetFlow (NF) datasets, every record gets a random timestamp: a random number of seconds in [0, 86400). Records are then sorted by these timestamps. | `Data/datasets_NF.py` L57–L71, L105, L120 |
| 2 | Edge timestamps are used as the filtration values for persistent homology. | `Spatial-Frequency Encoders/Spatial Encoder_NAPH.py` L48–L52 |
| 3 | The persistence diagram is converted to a fixed persistence image. It is not learned. | `Spatial Encoder_NAPH.py` L52, L65 (persim `PersistenceImager`) |
| 4 | The CIC datasets use their real timestamps. | `Data/datasets_CIC.py` L56–L57, L74–L76 |
| 5 | The script builds only a training loader and evaluates on it. Metrics are computed on the full stream after `best_map` matches clusters to the true labels. | `main_Intrusion Detector.py` L76, L353, L384–L399 |

**Impact**
- On the NF datasets, the topological filtration orders edges at random. Its PH features therefore carry no temporal information, and any benefit reported on those datasets cannot be attributed to temporal topology.
- Evaluating clustering on the same stream it was fitted on, with label matching, is common for unsupervised clustering. It is still not a held-out detection test.
- The separate "Online" script was not checked.

**Suggested wording:**
> "In the released code (commit bb8ccf5), records in the NetFlow-v2 datasets are assigned random timestamps (datasets_NF.py L57–L71), which then serve as the edge filtration for the persistence computation (Spatial Encoder_NAPH.py L48–L52)."

---

**Note (tdaad @ `d103a2c`, 19 Nov 2026):** the README quickstart (`np.random.randn(1000, 3)`) fails with any gudhi version (tested 3.11.0 and 3.13.0), because tdaad uses the sensors as points and 3 sensors can't form an H1 loop, so Atol gets an empty H1 diagram; with 38 channels (SMD width) it runs.

---

## 4. Fluent Bit `tda` processor (@ `25ea7e1`, found 19 Nov 2026 while building F5)

Not a re-test of a paper; a finding about what the processor outputs. Files: `tda.c`, `src/ripser/flb_ripser_wrapper.cpp`, `lib/ripser-1.2.1/ripser.cpp` (all at the pinned commit, copies in `external/fluent-bit/`).

- **What the processor reports as `betti0/1/2` is not a Betti number.** The wrapper's comment (L70) says Betti_k is "the number of infinite bars in dimension k". Ripser emits an infinite bar with `death = -1` (`ripser.cpp` L618, L789), and the wrapper's callback returns early when `death <= birth` (`flb_ripser_wrapper.cpp` L90). So infinite bars are never counted. The code counts **finite** bars (death at or below the chosen threshold) whose persistence is at least `1e-3` (L96–97, an absolute constant, not scaled to the data).
- **Consequence:** a loop that is still alive at the chosen distance threshold is not counted. The number is also in the units of the input (log1p of a rate), because of the absolute `1e-3` cut.
- **Scale scan** (`tda.c` L807, L967–L995): nine quantiles (0.1–0.9) of the pairwise distances are tried; the scale with the most H1 bars wins, and only if no scale has H1 bars does the largest beta0 win. The reported (beta0, beta1, beta2) come from different scales for different windows.
- **No evaluation** of any of this is published (PR fluent/fluent-bit#11250). Our F5 copies the code, not the comment (`src/features_fluentbit.py`).
- Wording for the paper: "the released code counts finite bars above 1e-3, whereas its comment describes infinite bars".

---

## Rules for the paper

- Always cite the commit and lines.
- Say "the released code"; never say "the authors did".
- Do not say that results are invalid. Say what the code does and what that implies.
- Before submission, re-check each repository for newer commits. If one is fixed, report both versions.

---

## Optional email to the authors

Send one per repository and fill in the brackets.

> **Subject:** Question about the evaluation code in [repo name]
>
> Dear [author],
>
> I'm preparing a review of topological data analysis for security and systems telemetry, and your paper "[title]" is one of the works covered. While reading the released code (commit [hash]), I noticed that [one-sentence description, e.g. "the threshold falls back to the test-set anomaly rate when the validation split has no anomalies", experiment_core.py L105–L112, L216].
>
> Could you confirm whether the results reported in the paper used this code path, or a different protocol? I want to describe your work accurately and will cite any clarification you provide.
>
> Thank you for releasing the code.
>
> Best regards,
> [Name, affiliation]
