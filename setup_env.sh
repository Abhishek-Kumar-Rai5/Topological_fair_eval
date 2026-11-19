#!/usr/bin/env bash
# Rebuild the project environment from scratch (Python 3.11, pinned versions).
# Needs uv: https://docs.astral.sh/uv/  (tested with uv 0.13.8)
set -euo pipefail
cd "$(dirname "$0")"

uv python install 3.11
rm -rf .venv
uv venv --python 3.11 .venv

# 1. All pinned packages (compiled from requirements.in).
uv pip sync --python .venv/bin/python requirements.txt

# 2. tdaad: declares requires-python>=3.12 but runs on 3.11, so install it
#    without the version check and without re-resolving its deps (already pinned above).
uv pip install --python .venv/bin/python pip==25.3
.venv/bin/python -m pip install --no-deps --ignore-requires-python -r requirements-nodeps.txt

# 3. Smoke test: every library on toy data (no project data touched).
.venv/bin/python scripts/check_env.py
