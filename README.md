# policy-eval-harness

`policy-eval-harness` is a standalone Python package for deterministic replay-driven policy evaluation in sequential systems. The current public core loads a prebuilt universe from CSV or Parquet, replays one or more Python-callable policy variants against the exact same ordered case set, and writes deterministic replay artifacts.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## CLI Help

```bash
policy-eval --help
```

## Replay Run

```bash
policy-eval replay run --manifest examples/replay-manifest.yaml --out-dir ./artifacts
```

The replay manifest locks three public inputs:

- `universe`: `cases_path`, `steps_path`
- `executor`: `import_path`, optional `params`
- `variants`: `path`
- `run`: `variant_ids`

The command writes:

- `episode_summary.csv`
- `step_trace.parquet`
- `run_metadata.json`
