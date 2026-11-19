# topo-fair-eval

Audit and fair re-evaluation of TDA anomaly detectors for systems and security telemetry. See CLAUDE.md and preregistration/.

## Environment

Python 3.11, managed with [uv](https://docs.astral.sh/uv/) (tested with uv 0.13.8).

```bash
./setup_env.sh            # creates .venv, installs pinned deps, runs scripts/check_env.py
source .venv/bin/activate
```

- `requirements.in`: top-level dependencies. `requirements.txt`: full pinned lock, compiled with
  `uv pip compile requirements.in --python-version 3.11 -o requirements.txt`.
- `requirements-nodeps.txt`: `tdaad`, pinned to commit `d103a2c` (v1.6.0). It declares Python >= 3.12,
  but its code runs on 3.11, so it is installed with `--no-deps --ignore-requires-python`.
  Its dependencies are pinned in `requirements.txt`.
