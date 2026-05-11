from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import pandas as pd

from policy_eval_harness import __version__
from policy_eval_harness._utils.json import canonical_json, normalize_json_value
from policy_eval_harness.replay.constants import SUMMARY_COLUMNS, TRACE_COLUMNS
from policy_eval_harness.replay.manifest import load_replay_manifest, resolve_callable
from policy_eval_harness.replay.types import (
    ExecutorConfig,
    ReplayArtifacts,
    ReplayCase,
    ReplayManifest,
    ReplayStep,
    ReplayStepInput,
    ReplayTransition,
    ReplayUniverse,
    VariantConfig,
)
from policy_eval_harness.replay.universe import load_universe


def run_replay_from_manifest(manifest_path: Path, out_dir: Path) -> ReplayArtifacts:
    manifest = load_replay_manifest(Path(manifest_path))
    universe = load_universe(manifest.universe.cases_path, manifest.universe.steps_path)
    variant_map = {variant.variant_id: variant for variant in manifest.variants}
    selected_variants = [variant_map[variant_id] for variant_id in manifest.run_variant_ids]
    executor_callable = resolve_callable(manifest.executor.import_path)
    return run_replay(manifest, universe, selected_variants, executor_callable, out_dir)


def run_replay(
    manifest: ReplayManifest,
    universe: ReplayUniverse,
    variants: Sequence[VariantConfig],
    executor_callable: Any,
    out_dir: Path,
) -> ReplayArtifacts:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    trace_rows = []

    for variant in variants:
        policy_callable = resolve_callable(variant.policy_import_path)
        for case in universe.cases:
            summary_row, case_trace_rows = _run_case(
                case=case,
                steps=universe.steps_by_case.get(case.case_id, ()),
                variant=variant,
                policy_callable=policy_callable,
                executor_callable=executor_callable,
                executor_config=manifest.executor,
            )
            summary_rows.append(summary_row)
            trace_rows.extend(case_trace_rows)

    episode_summary_path = out_dir / manifest.output.episode_summary
    step_trace_path = out_dir / manifest.output.step_trace
    run_metadata_path = out_dir / manifest.output.run_metadata

    _write_csv(episode_summary_path, SUMMARY_COLUMNS, summary_rows)
    _write_parquet(step_trace_path, TRACE_COLUMNS, trace_rows)
    _write_run_metadata(run_metadata_path, manifest, universe, variants)
    return ReplayArtifacts(
        episode_summary_path=episode_summary_path,
        step_trace_path=step_trace_path,
        run_metadata_path=run_metadata_path,
    )


def _run_case(
    case: ReplayCase,
    steps: Sequence[ReplayStep],
    variant: VariantConfig,
    policy_callable: Any,
    executor_callable: Any,
    executor_config: ExecutorConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    trace_rows = []
    state = None
    decision_count = 0
    error_count = 0
    summary_metrics = {}
    final_status = None
    termination_reason = None
    terminated = False
    n_steps = 0
    start_ts = case.start_ts_utc
    end_ts = case.start_ts_utc

    for step in steps:
        if start_ts is None:
            start_ts = step.ts_utc
        end_ts = step.ts_utc
        step_input = ReplayStepInput(
            variant_id=variant.variant_id,
            case=case,
            step=step,
            state=normalize_json_value(state),
        )
        state_before = normalize_json_value(state)

        try:
            action = policy_callable(step_input, variant.policy_params)
        except Exception as exc:
            n_steps += 1
            error_count += 1
            final_status = "error"
            termination_reason = "policy_exception"
            summary_metrics = _merge_metrics(summary_metrics, {"error_stage": "policy"})
            trace_rows.append(
                _build_trace_row(case.case_id, variant.variant_id, step, step.observation, None, state_before, None, None, None, "policy_exception: {}".format(exc))
            )
            break

        action = normalize_json_value(action)
        if action is not None:
            decision_count += 1

        try:
            transition = executor_callable(step_input, action, executor_config.params)
        except Exception as exc:
            n_steps += 1
            error_count += 1
            final_status = "error"
            termination_reason = "executor_exception"
            summary_metrics = _merge_metrics(summary_metrics, {"error_stage": "executor"})
            trace_rows.append(
                _build_trace_row(case.case_id, variant.variant_id, step, step.observation, action, state_before, None, None, None, "executor_exception: {}".format(exc))
            )
            break

        transition = _coerce_transition(transition)
        state = transition.next_state
        n_steps += 1
        final_status = transition.final_status or final_status or "completed"
        if transition.termination_reason is not None:
            termination_reason = transition.termination_reason
        summary_metrics = _merge_metrics(summary_metrics, transition.metrics)
        trace_rows.append(
            _build_trace_row(
                case.case_id,
                variant.variant_id,
                step,
                step.observation,
                action,
                state_before,
                transition.next_state,
                transition.metrics,
                transition.trace,
                "",
            )
        )
        if transition.terminated:
            terminated = True
            break

    if final_status is None:
        final_status = "no_steps" if not steps else "completed"
    if start_ts is None and steps:
        start_ts = steps[0].ts_utc
    if end_ts is None and steps:
        end_ts = steps[-1].ts_utc

    return (
        {
            "case_id": case.case_id,
            "variant_id": variant.variant_id,
            "start_ts_utc": start_ts,
            "end_ts_utc": end_ts,
            "n_steps": n_steps,
            "terminated": terminated,
            "termination_reason": termination_reason,
            "final_status": final_status,
            "decision_count": decision_count,
            "error_count": error_count,
            "summary_metrics_json": canonical_json(summary_metrics),
        },
        trace_rows,
    )


def _build_trace_row(
    case_id: str,
    variant_id: str,
    step: ReplayStep,
    observation: Mapping[str, Any],
    action: Any,
    state_before: Any,
    state_after: Any,
    transition_metrics: Any,
    trace_payload: Any,
    error: str,
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "variant_id": variant_id,
        "step_index": step.step_index,
        "ts_utc": step.ts_utc,
        "observation_json": canonical_json(observation),
        "action_json": canonical_json(action),
        "state_before_json": canonical_json(state_before),
        "state_after_json": canonical_json(state_after),
        "transition_metrics_json": canonical_json(transition_metrics or {}),
        "trace_json": canonical_json(trace_payload or {}),
        "error": error,
    }


def _merge_metrics(existing: MutableMapping[str, Any], new_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in normalize_json_value(dict(new_metrics)).items():
        current_value = merged.get(key)
        if _is_numeric(current_value) and _is_numeric(value):
            merged[key] = current_value + value
        else:
            merged[key] = value
    return merged


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_transition(transition: Any) -> ReplayTransition:
    if isinstance(transition, ReplayTransition):
        return ReplayTransition(
            next_state=normalize_json_value(transition.next_state),
            terminated=bool(transition.terminated),
            termination_reason=transition.termination_reason,
            final_status=transition.final_status,
            metrics=normalize_json_value(transition.metrics) or {},
            trace=normalize_json_value(transition.trace) or {},
        )
    if not isinstance(transition, Mapping):
        raise TypeError("Executor must return ReplayTransition or a mapping.")
    return ReplayTransition(
        next_state=normalize_json_value(transition.get("next_state")),
        terminated=bool(transition.get("terminated", False)),
        termination_reason=transition.get("termination_reason"),
        final_status=transition.get("final_status"),
        metrics=normalize_json_value(transition.get("metrics", {})) or {},
        trace=normalize_json_value(transition.get("trace", {})) or {},
    )


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _write_parquet(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    pd.DataFrame(rows, columns=columns).to_parquet(path, index=False)


def _write_run_metadata(
    path: Path,
    manifest: ReplayManifest,
    universe: ReplayUniverse,
    variants: Sequence[VariantConfig],
) -> None:
    metadata = {
        "artifact_files": {
            "episode_summary": manifest.output.episode_summary,
            "run_metadata": manifest.output.run_metadata,
            "step_trace": manifest.output.step_trace,
        },
        "manifest_hash": manifest.manifest_hash,
        "package_version": __version__,
        "universe_row_counts": {
            "cases": universe.case_count,
            "steps": universe.step_count,
        },
        "variant_ids": [variant.variant_id for variant in variants],
    }
    path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")

