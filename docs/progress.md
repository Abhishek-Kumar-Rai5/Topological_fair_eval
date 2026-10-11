# Progress

Updated at the end of every task. Latest entry first.


## 2026-11-19 (later): F3 decisions applied, CTU-13 scenario 10 excluded, CTU-13 facts

**Done**
- F3 rewritten per your decisions: z-score with training mean/std (std floor 0.01), Betti scales 0.2–2.0 × the training median distance (once per machine), finite bars only. Tests updated: 50 pass. Real-data check (2 machines): all finite, about 1.2 ms per window.
- `docs/implementation_notes.md` F3 section updated (choices 12–15 now confirmed). `deviations_log.md` has 4 F3 clarifications and the scenario 10 exclusion.
- `scripts/ctu13_facts.py` writes `docs/ctu13_facts.md` (label strings, infected IPs, normal hosts per scenario).

**Findings (CTU-13)**
- 12 usable scenarios. 1 infected IP in most scenarios, 10 in scenario 9, 3 in scenarios 11 and 12.
- Only 5–7 normal hosts per scenario, and only 7 distinct normal IPs across all 12 scenarios (4 appear in every scenario). Normal-host windows are therefore not independent across scenarios, which limits E4.

**Next**
1. Step 11: F5 Fluent Bit-style features from `external/fluent-bit/tda.c`.
2. Runner script (config → results, with run metadata), then E2 and E3 on SMD.

**Open decisions (Anirudh)**
- Behaviour types for E4 (prereg §2): the botnet label strings name protocols or traffic types (DNS, SPAM, CC1, HTTP-Ad, Binary-Download), not single behaviours. Decide whether to use them as behaviours or to give each scenario a type from its README (Mixed if they disagree). Log it before E4 runs.
- Scenario 12 and 3 have From-Botnet flows sent by non-listed IPs; the plan treats only README-listed IPs as infected hosts. Confirm.
- Earlier items still open: fill TBD dates in `docs/decision_log.md`; check the `prereg-v3` tag is on the remote; Step 4 self re-check.

## 2026-11-19: status check after the laptop shutdown

**Done**
- Checked the repo state. `pytest -q`: 46 passed. `scripts/check_env.py`: ALL OK.
- Found the git repo damaged: the Step 10 commit `f458c4a` and 6 other loose objects are empty (0 bytes). The files on disk are intact. Last healthy commit: `0cf1360`.
- Plan status: Steps 1, 3, 6, 7, 8, 9 done. Step 10 code and tests done, but its commit was lost. Steps 11–21 not started. Steps 2 (push), 4 and 5 (dates) are partly or fully yours.

**Next**
1. Repair git: delete the empty objects, point `master` back at `0cf1360`, `git reset`, then recommit the 4 Step 10 files (commands given in chat).
2. Step 11: F5 Fluent Bit-style β0/β1/β2 features.
3. Runner script from config to results (with run metadata), then E2 and E3 on SMD.

**Open decisions (Anirudh)**
- Confirm or change F3 choices 12–15 in `docs/implementation_notes.md`. A change after results exist goes in `deviations_log.md`.
- CTU-13 scenario 10 (capture 51) has no `.binetflow`, so only 12 of 13 scenarios are usable. Decide how E4 handles this (needs a `deviations_log.md` entry if it differs from the prereg).
- Fill the TBD dates in `docs/decision_log.md`.
- Check that the tag `prereg-v3` is on the remote (`git ls-remote --tags origin`). `origin/master` is still at `773d00d`.
- Step 4: self re-check of 7 papers.
