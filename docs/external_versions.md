# External code: pinned versions

Recorded on 2026-11-19. Third-party code lives in `external/` (gitignored), except `tdaad`, which is installed into `.venv` (see `requirements-nodeps.txt`).

| Name | Repository | Commit | Commit date | Licence | Local copy |
|---|---|---|---|---|---|
| TAME | https://github.com/SPIRE-Lab-2025/TAME | `e77c0d93434bdd6b20db9db60e524a30a20a35b5` | 2026-07-18 | **None.** No LICENSE file; the README's License section is an empty placeholder | `external/TAME/` |
| PHANTOM | https://github.com/pnnl/PHANTOM | `8caa25a8a5e305d2b2ec91bfdd9a743f52034b70` | 2023-03-14 | BSD (text in README; © 2022 Battelle Memorial Institute). No LICENSE file, so GitHub shows none | `external/PHANTOM/` |
| Fluent Bit `tda` processor | https://github.com/fluent/fluent-bit (`plugins/processor_tda/`) | `25ea7e125fd2fc4a8466a67255a5d88ca760ac14` (latest commit touching `tda.c` on `master` as of 2026-11-19: "processor_tda: add TDA metrics processor based on ripser") | 2026-02-14 | Apache-2.0 | `external/fluent-bit/tda.c` (SHA-256 `b998b6c4af2ca1fd050b521a23ceb5dd729f2ac65b9712b98976ca449a1b848e`), plus `tda.h` from the same commit |
| tdaad (TADA) | https://github.com/IRT-SystemX/tdaad | `d103a2cb5b48ec435afff9b7f75185ff2da6ca3f` (tag `1.6.0-rc`, package version 1.6.0) | 2026-02-03 | MPL-2.0 | installed in `.venv` |

Extra Fluent Bit files at the same commit (needed to read what `tda.c` calls; fetched 2026-11-19 from raw.githubusercontent.com, gitignored in `external/fluent-bit/`):

| File | SHA-256 |
|---|---|
| `src/ripser/flb_ripser_wrapper.cpp` | `d625729c803b6969cc41cc4f5f2265561e29104d4166767be1bd4de06ec76691` |
| `include/fluent-bit/ripser/flb_ripser_wrapper.h` | `cfd8d24a985f6ea2604620dc695f410dd94c453e0f9067e9a6aebbecbc102e2c` |
| `lib/ripser-1.2.1/ripser_internal.hpp` | `3db542336f650894a3ffb1ddefb8928761ee7ed6832baa8e15584ca8ac2ac89a` |
| `lib/ripser-1.2.1/ripser.cpp` | `054ea76ce2d2a2ae2985090f65f5980de6051654687922b3acec5e84f8927024` |

## Notes

- **TAME has no licence.** We can run it locally for research, but we must not redistribute its code. `external/` is gitignored, so nothing is redistributed. Results and short quoted line references are fine to report.
- **tdaad** declares `requires-python >= 3.12`. It is installed on Python 3.11 with `--ignore-requires-python --no-deps`. The code parses under 3.11, and the environment check (`scripts/check_env.py`) passes.
- The commits for TAME and PHANTOM match the ones used in `docs/code_verification.md` and `CLAUDE.md` §4.
