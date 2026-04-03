from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from policy_eval_harness import __version__
from policy_eval_harness.replay.types import (
    ExecutorConfig,
    OutputConfig,
    ReplayArtifacts,
    ReplayCase,
    ReplayManifest,
    ReplayStep,
    ReplayStepInput,
    ReplayTransition,
    ReplayUniverse,
    UniverseConfig,
    VariantConfig,
)

SUMMARY_COLUMNS = [
    "case_id",
    "variant_id",
    "start_ts_utc",
    "end_ts_utc",
    "n_steps",
    "terminated",
    "termination_reason",
    "final_status",
    "decision_count",
    "error_count",
    "summary_metrics_json",
]

TRACE_COLUMNS = [
    "case_id",
    "variant_id",
    "step_index",
    "ts_utc",
    "observation_json",
    "action_json",
    "state_before_json",
    "state_after_json",
    "transition_metrics_json",
    "trace_json",
    "error",
]

CASE_RESERVED_COLUMNS = {"case_id", "start_ts_utc"}
STEP_RESERVED_COLUMNS = {"case_id", "step_index", "ts_utc"}


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize_json_value(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def normalize_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return normalize_json_value(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return _normalize_timestamp(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


def load_replay_manifest(manifest_path: Path) -> ReplayManifest:
    manifest_path = manifest_path.resolve()
    manifest_data = _load_mapping(manifest_path)
    base_dir = manifest_path.parent

    universe_data = _require_mapping(manifest_data, "universe")
    executor_data = _require_mapping(manifest_data, "executor")
    variants_data = _require_mapping(manifest_data, "variants")
    run_data = _require_mapping(manifest_data, "run")
    output_data = manifest_data.get("output", {})
    if output_data is None:
        output_data = {}
    if not isinstance(output_data, Mapping):
        raise ValueError("Manifest field 'output' must be a mapping when provided.")

    universe = UniverseConfig(
        cases_path=_resolve_path(base_dir, _require_string(universe_data, "cases_path")),
        steps_path=_resolve_path(base_dir, _require_string(universe_data, "steps_path")),
    )
    executor = ExecutorConfig(
        import_path=_require_string(executor_data, "import_path"),
        params=_normalize_mapping(executor_data.get("params", {}), "executor.params"),
    )
    variants = _load_variants(_resolve_path(base_dir, _require_string(variants_data, "path")))

    run_variant_ids = tuple(_require_string_list(run_data, "variant_ids"))
    if not run_variant_ids:
        raise ValueError("Manifest field 'run.variant_ids' must not be empty.")
    if len(set(run_variant_ids)) != len(run_variant_ids):
        raise ValueError("Manifest field 'run.variant_ids' contains duplicates.")

    variant_map = {variant.variant_id: variant for variant in variants}
    missing_variant_ids = [variant_id for variant_id in run_variant_ids if variant_id not in variant_map]
    if missing_variant_ids:
        raise ValueError(
            "Variant manifest is missing requested variants: " + ", ".join(sorted(missing_variant_ids))
        )

    output = OutputConfig(
        episode_summary=_optional_string(output_data, "episode_summary") or "episode_summary.csv",
        step_trace=_optional_string(output_data, "step_trace") or "step_trace.parquet",
        run_metadata=_optional_string(output_data, "run_metadata") or "run_metadata.json",
    )

    canonical_manifest = {
        "executor": {
            "import_path": executor.import_path,
            "params": executor.params,
        },
        "output": asdict(output),
        "run": {
            "variant_ids": list(run_variant_ids),
        },
        "universe": {
            "cases_path": _require_string(universe_data, "cases_path"),
            "steps_path": _require_string(universe_data, "steps_path"),
        },
        "variants": [
            {
                "policy": {
                    "import_path": variant.policy_import_path,
                    "params": variant.policy_params,
                },
                "variant_id": variant.variant_id,
            }
            for variant in variants
        ],
    }
    manifest_hash = hashlib.sha256(canonical_json(canonical_manifest).encode("utf-8")).hexdigest()

    return ReplayManifest(
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        universe=universe,
        executor=executor,
        variants=variants,
        run_variant_ids=run_variant_ids,
        output=output,
    )


def load_universe(cases_path: Path, steps_path: Path) -> ReplayUniverse:
    case_records = _read_table(cases_path)
    step_records = _read_table(steps_path)

    cases = []
    seen_case_ids = set()
    for record in case_records:
        case_id = _normalize_case_id(record.get("case_id"))
        if case_id in seen_case_ids:
            raise ValueError("Duplicate case_id in cases table: {!r}".format(case_id))
        seen_case_ids.add(case_id)
        metadata = {
            key: normalize_json_value(value)
            for key, value in record.items()
            if key not in CASE_RESERVED_COLUMNS
        }
        cases.append(
            ReplayCase(
                case_id=case_id,
                start_ts_utc=_optional_timestamp(record.get("start_ts_utc")),
                metadata=metadata,
            )
        )
    cases.sort(key=lambda case: case.case_id)

    steps_by_case = defaultdict(list)
    for record in step_records:
        case_id = _normalize_case_id(record.get("case_id"))
        if case_id not in seen_case_ids:
            raise ValueError("Step row references unknown case_id: {!r}".format(case_id))
        steps_by_case[case_id].append(
            ReplayStep(
                case_id=case_id,
                step_index=_normalize_step_index(record.get("step_index")),
                ts_utc=_required_timestamp(record.get("ts_utc"), "steps.ts_utc"),
                observation={
                    key: normalize_json_value(value)
                    for key, value in record.items()
                    if key not in STEP_RESERVED_COLUMNS
                },
                metadata={},
            )
        )

    ordered_steps = {}
    for case in cases:
        ordered_steps[case.case_id] = tuple(
            sorted(
                steps_by_case.get(case.case_id, []),
                key=lambda step: (step.case_id, step.step_index, step.ts_utc),
            )
        )

    return ReplayUniverse(cases=tuple(cases), steps_by_case=ordered_steps)


def resolve_callable(import_path: str) -> Any:
    module_name, attribute_path = _split_import_path(import_path)
    module = importlib.import_module(module_name)
    target = module
    for attribute_name in attribute_path.split("."):
        target = getattr(target, attribute_name)
    if not callable(target):
        raise TypeError("Resolved import path is not callable: {!r}".format(import_path))
    return target


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
                _build_trace_row(
                    case.case_id,
                    variant.variant_id,
                    step,
                    step.observation,
                    None,
                    state_before,
                    None,
                    None,
                    None,
                    "policy_exception: {}".format(exc),
                )
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
                _build_trace_row(
                    case.case_id,
                    variant.variant_id,
                    step,
                    step.observation,
                    action,
                    state_before,
                    None,
                    None,
                    None,
                    "executor_exception: {}".format(exc),
                )
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


def _load_variants(variants_path: Path) -> Tuple[VariantConfig, ...]:
    raw_variants = _load_structured_file(variants_path)
    if isinstance(raw_variants, Mapping):
        raw_variants = raw_variants.get("variants")
    if not isinstance(raw_variants, list):
        raise ValueError("Variant manifest must be a list or a mapping containing 'variants'.")

    variants = []
    seen_variant_ids = set()
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, Mapping):
            raise ValueError("Each variant entry must be a mapping.")
        variant_id = _require_string(raw_variant, "variant_id")
        if variant_id in seen_variant_ids:
            raise ValueError("Duplicate variant_id in variant manifest: {!r}".format(variant_id))
        seen_variant_ids.add(variant_id)
        policy_data = _require_mapping(raw_variant, "policy")
        variants.append(
            VariantConfig(
                variant_id=variant_id,
                policy_import_path=_require_string(policy_data, "import_path"),
                policy_params=_normalize_mapping(policy_data.get("params", {}), "policy.params"),
            )
        )
    return tuple(variants)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    raw_data = _load_structured_file(path)
    if not isinstance(raw_data, Mapping):
        raise ValueError("Structured file must contain a mapping: {!r}".format(str(path)))
    return raw_data


def _load_structured_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_table(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported table format: {!r}".format(path.suffix))
    return frame.to_dict(orient="records")


def _require_mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError("Manifest field '{}' must be a mapping.".format(key))
    return value


def _require_string(container: Mapping[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Manifest field '{}' must be a non-empty string.".format(key))
    return value


def _optional_string(container: Mapping[str, Any], key: str) -> Optional[str]:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Manifest field '{}' must be a non-empty string when provided.".format(key))
    return value


def _require_string_list(container: Mapping[str, Any], key: str) -> List[str]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ValueError("Manifest field '{}' must be a list.".format(key))
    output = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Manifest field '{}' must contain only non-empty strings.".format(key))
        output.append(item)
    return output


def _normalize_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Field '{}' must be a mapping when provided.".format(field_name))
    return {str(key): normalize_json_value(item) for key, item in value.items()}


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _split_import_path(import_path: str) -> Tuple[str, str]:
    if ":" in import_path:
        module_name, attribute_path = import_path.split(":", 1)
    else:
        module_name, separator, attribute_path = import_path.rpartition(".")
        if not separator:
            raise ValueError("Import path must use 'module:attribute' or 'module.attribute'.")
    if not module_name or not attribute_path:
        raise ValueError("Import path must use 'module:attribute' or 'module.attribute'.")
    return module_name, attribute_path


def _normalize_case_id(value: Any) -> str:
    normalized = normalize_json_value(value)
    if normalized in (None, ""):
        raise ValueError("Cases and steps require a non-empty case_id.")
    return str(normalized)


def _normalize_step_index(value: Any) -> int:
    normalized = normalize_json_value(value)
    if normalized is None or isinstance(normalized, bool):
        raise ValueError("Step rows require an integer step_index.")
    if isinstance(normalized, float) and not normalized.is_integer():
        raise ValueError("Step rows require an integer step_index.")
    return int(normalized)


def _optional_timestamp(value: Any) -> Optional[str]:
    if normalize_json_value(value) is None:
        return None
    return _required_timestamp(value, "timestamp")


def _required_timestamp(value: Any, field_name: str) -> str:
    normalized = _normalize_timestamp(value)
    if normalized is None:
        raise ValueError("Field '{}' requires a valid UTC timestamp.".format(field_name))
    return normalized


def _normalize_timestamp(value: Any) -> Optional[str]:
    normalized = normalize_json_value(value)
    if normalized is None:
        return None
    timestamp = pd.Timestamp(normalized)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    python_datetime = timestamp.to_pydatetime().astimezone(timezone.utc)
    iso_value = python_datetime.isoformat().replace("+00:00", "Z")
    if "." not in iso_value:
        return iso_value
    prefix, suffix = iso_value.split(".", 1)
    fraction = suffix[:-1].rstrip("0")
    return prefix + ("." + fraction if fraction else "") + "Z"
