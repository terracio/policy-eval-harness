# Publication Readiness

Release target: `v0.1.0`  
Verdict: `ready`

This checklist captures the final release gate for the public `policy-eval-harness` repo. It is intentionally operational rather than narrative.

## Gates

- `[pass]` Naming and domain-neutrality
  Checked module names, example names, CLI help, and docs. Public materials stay on the methodology side of the boundary and do not use private trading identifiers.

- `[pass]` Secrets and credentials
  Automated forbidden-pattern scan passed against the public repo tree for the release list:
  `PENGU`, `Bybit`, `Binance`, `live_v1`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `BYBIT_`, `BINANCE_`, `service_account`, `.env`, `postgresql:///`, `artifacts/ml`.
  Audit/control files that intentionally contain the scan list are excluded from content scanning:
  `docs/publication_readiness.md`, `scripts/publication_readiness_check.py`.

- `[pass]` Artifact safety
  Checked-in runnable assets are limited to the public demo and workflow examples under `examples/`. No live configs, provider secrets, trained private artifacts, notebooks, or prompt source files are present.

- `[pass]` Docs and link hygiene
  Verified local Markdown links across `README.md`, `docs/`, and example READMEs. Release metadata and public command surface are aligned.

- `[pass]` Runnable fresh-clone verification
  Release scripts are designed to run from a fresh editable install with the installed `policy-eval` console script. `scripts/verify_public_contracts.py` drives:
  `policy-eval demo run`
  `policy-eval workflow label-compare run`
  `policy-eval workflow ablation-2x2 run`
  and compares generated outputs to the checked-in goldens.

- `[pass]` Reproducibility and golden verification
  `scripts/verify_public_contracts.py` regenerates demo and workflow outputs twice and compares normalized CSV, JSON, Markdown, and Parquet artifacts against the checked-in goldens and against a second rerun.

- `[pass]` License, version, and release metadata
  `LICENSE` is MIT. Package metadata and `__version__` are pinned to `0.1.0`. First release note lives at `docs/releases/v0.1.0.md`.

## Checks Run

- `python -m unittest discover -s tests -t .`
- `python scripts/verify_public_contracts.py`
- `python scripts/publication_readiness_check.py`
- `git diff --check`

Local verification for this ticket was run from a fresh editable-install Python 3.12 venv on this machine. The repository CI remains pinned to Python 3.10 in `.github/workflows/ci.yml`.

## Blocking Findings

- None.

## Next Action

- Merge `codex/cods-37` into `main`.
- Tag `v0.1.0`.
- Make the GitHub repo public when you are ready to publish the first release.
