# Decision log

Dates marked **TBD** must be filled in from your chat history and notes. Do not guess them.

| Date | Idea or decision | Evidence checked | Decision | Reason |
|---|---|---|---|---|
| TBD (Oct 2026) | TDA for neural-network trojan detection (TopoTrojDetection) | Paper, supplement, code at commit 2e840e1, model archives | Rejected as an empirical project | Model archives unavailable (HTTP 429); bugs in the label loader and preprocessing; close prior work (ArcGen, HTell) |
| TBD | Path-conditioned PH on microservice traces (RCAEval) | TraceAnomaly, TraceVAE, TraceCRL, ChainLSTM | Not pursued | Path-aware latency modelling is already established; novelty too narrow |
| TBD | Workload-residual PH on Spark (Exathlon) | TADA, DIVAD | Not pursued | Both ingredients already exist; fixed centring leaves correlations unchanged, so the idea risks being a no-op |
| TBD | PH-based trace sampling | TraceMesh, TraceCRL | Not pursued | Too close to clustering-based sampling |
| TBD | HPC application-transfer study (Prodigy) | 2025 spectral-topological Deep SVDD paper for HPC | Not pursued | Topic already occupied |
| ≤ 2026-11-12 | Literature audit V1–V3 (76 records; search cutoff 14 Oct) | Audit document | Basis for a review | The evidence base is weak, so auditing it is itself a contribution |
| 2026-11-12 | Critical review of TDA for security and systems telemetry | Davies 2022, Georgiou 2026, 119 logged queries | Adopted | No critical, evaluation-focused review exists |
| 2026-11-13 | Code-level verification of TAME, TopoGDN and MSF-IDS | Repos at pinned commits | Adopted | Problems tied to exact lines (docs/code_verification.md) |
| 2026-11-13 → 11-16 | Search rounds 2 and 3; rubric scoring; forward citations | records CSVs, search logs | Completed | 95 records screened, 33 core papers (32 after D03 was excluded) |
| 2026-11-15 | One combined review (security plus systems) rather than separate ones | Systems literature alone has about 8 papers | Adopted | Systems alone is too thin |
| 2026-11-16 | SoK format: review plus a small fair experiment; prereg v2 | Dataset feasibility checks | Adopted, then revised | A review of weak papers alone would be weak |
| 2026-11-17 | Audit plus fair re-evaluation using released code and public data; PHANTOM and the zigzag re-test added | Repo and licence checks (PHANTOM BSD, tdaad MPL-2.0) | Adopted | The evidence must come from our own fair tests, not from the papers' claims |
| 2026-11-19 | Prereg v3: SMD replaces Exathlon; CTU-13 per-host windows; swap test; E5 injections; fixed-default detectors | Flaws found in v2 | Adopted | v2's CTU-13 labelling would break the experiment, and SMD validation data has no anomalies |
| 2026-11-19 | D03 excluded from the core set | Paper reports no Mapper output on any data | Adopted | Fails inclusion rule 1 in practice |
