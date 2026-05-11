from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from policy_eval_harness._utils.paths import resolve_path
from policy_eval_harness._utils.manifest import (
    optional_bool,
    reject_duplicate_strings,
    reject_unknown_keys,
    require_existing_path,
    require_numeric_mapping_values,
)
from policy_eval_harness.evaluation.constants import (
    DEFAULT_SELECTION_THRESHOLDS,
    DEFAULT_SEQUENTIAL_THRESHOLDS,
    PROFILE_REPLAY,
    PROFILE_SELECTION,
    SELECTION_GATES,
    SEQUENTIAL_GATES,
)
from policy_eval_harness.evaluation.types import (
    BootstrapConfig,
    EvaluationManifest,
    GateConfig,
    SplitConfig,
)
from policy_eval_harness.replay import canonical_json
from policy_eval_harness._utils.json import normalize_json_value


def load_evaluation_manifest(manifest_path: Path) -> EvaluationManifest:
    manifest_path = manifest_path.resolve()
    raw_manifest = _load_mapping(manifest_path)
    base_dir = manifest_path.parent
    reject_unknown_keys(
        raw_manifest,
        {"profile", "inputs", "baseline_variant_id", "candidate_variant_ids", "splits", "gates", "bootstrap"},
        "Evaluation manifest",
    )

    profile_id = _require_string(raw_manifest, "profile")
    if profile_id not in {PROFILE_REPLAY, PROFILE_SELECTION}:
        raise ValueError(f"Unsupported evaluation profile: {profile_id!r}")

    inputs_raw = _require_mapping(raw_manifest, "inputs")
    baseline_variant_id = _require_string(raw_manifest, "baseline_variant_id")
    candidate_variant_ids = tuple(_require_string_list(raw_manifest, "candidate_variant_ids"))
    if not candidate_variant_ids:
        raise ValueError("Manifest field 'candidate_variant_ids' must not be empty.")
    reject_duplicate_strings(candidate_variant_ids, "Manifest field 'candidate_variant_ids'")
    if baseline_variant_id in candidate_variant_ids:
        raise ValueError("Manifest field 'baseline_variant_id' must not also appear in 'candidate_variant_ids'.")

    splits_raw = _optional_mapping(raw_manifest, "splits")
    gates_raw = _optional_mapping(raw_manifest, "gates")
    bootstrap_raw = _optional_mapping(raw_manifest, "bootstrap")
    reject_unknown_keys(splits_raw, {"mode", "field", "holdout_cutoff_utc", "holdout_pct"}, "Manifest field 'splits'")
    reject_unknown_keys(gates_raw, {"profile", "thresholds"}, "Manifest field 'gates'")
    reject_unknown_keys(bootstrap_raw, {"enabled", "n_samples", "seed"}, "Manifest field 'bootstrap'")

    _validate_profile_inputs(profile_id, inputs_raw)
    inputs = _normalize_inputs(base_dir, inputs_raw)
    splits = SplitConfig(
        mode=_optional_string(splits_raw, "mode") or "existing",
        field=_optional_string(splits_raw, "field"),
        holdout_cutoff_utc=_optional_string(splits_raw, "holdout_cutoff_utc"),
        holdout_pct=_optional_float(splits_raw, "holdout_pct"),
    )
    _validate_splits(splits)
    default_gate_profile = SEQUENTIAL_GATES if profile_id == PROFILE_REPLAY else SELECTION_GATES
    gate_profile = _optional_string(gates_raw, "profile") or default_gate_profile
    if gate_profile != default_gate_profile:
        raise ValueError(
            f"Unsupported gates.profile {gate_profile!r} for profile {profile_id!r}; expected {default_gate_profile!r}."
        )
    thresholds = _normalize_mapping(gates_raw.get("thresholds", {}), "gates.thresholds")
    threshold_keys = _supported_threshold_keys(default_gate_profile)
    reject_unknown_keys(thresholds, threshold_keys, "gates.thresholds")
    require_numeric_mapping_values(thresholds, "gates.thresholds")
    gates = GateConfig(
        profile_id=gate_profile,
        thresholds=thresholds,
    )
    bootstrap = BootstrapConfig(
        enabled=optional_bool(bootstrap_raw.get("enabled"), "bootstrap.enabled", default=False),
        n_samples=int(bootstrap_raw.get("n_samples", 0) or 0),
        seed=int(bootstrap_raw.get("seed", 0) or 0),
    )
    if bootstrap.enabled and bootstrap.n_samples <= 0:
        raise ValueError("Bootstrap requires 'n_samples' > 0 when enabled.")

    manifest_hash = hashlib.sha256(canonical_json(_canonical_manifest(
        profile_id=profile_id,
        inputs=inputs,
        baseline_variant_id=baseline_variant_id,
        candidate_variant_ids=candidate_variant_ids,
        splits=splits,
        gates=gates,
        bootstrap=bootstrap,
    )).encode("utf-8")).hexdigest()

    return EvaluationManifest(
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        profile_id=profile_id,
        inputs=inputs,
        baseline_variant_id=baseline_variant_id,
        candidate_variant_ids=candidate_variant_ids,
        splits=splits,
        gates=gates,
        bootstrap=bootstrap,
    )


def _validate_profile_inputs(profile_id: str, inputs: Mapping[str, Any]) -> None:
    if profile_id == PROFILE_REPLAY:
        reject_unknown_keys(inputs, {"episode_summary_path", "cases_path"}, "Manifest field 'inputs'")
        _require_string(inputs, "episode_summary_path")
        return
    if profile_id == PROFILE_SELECTION:
        reject_unknown_keys(inputs, {"panel_path"}, "Manifest field 'inputs'")
        _require_string(inputs, "panel_path")
        return
    raise ValueError(f"Unsupported evaluation profile: {profile_id!r}")


def _validate_splits(splits: SplitConfig) -> None:
    if splits.mode == "existing":
        if splits.field or splits.holdout_cutoff_utc or splits.holdout_pct is not None:
            raise ValueError("splits.mode 'existing' does not accept field, holdout_cutoff_utc, or holdout_pct.")
        return
    if splits.mode != "time_holdout":
        raise ValueError(f"Unsupported splits.mode: {splits.mode!r}")
    has_cutoff = splits.holdout_cutoff_utc is not None
    has_pct = splits.holdout_pct is not None
    if has_cutoff == has_pct:
        raise ValueError("splits.mode 'time_holdout' requires exactly one of holdout_cutoff_utc or holdout_pct.")
    if splits.holdout_pct is not None and not 0 <= splits.holdout_pct <= 1:
        raise ValueError("splits.holdout_pct must be between 0 and 1.")


def _supported_threshold_keys(gate_profile: str) -> set[str]:
    if gate_profile == SEQUENTIAL_GATES:
        return set(DEFAULT_SEQUENTIAL_THRESHOLDS) | {"min_delta_mean_utility_ci_low"}
    if gate_profile == SELECTION_GATES:
        return set(DEFAULT_SELECTION_THRESHOLDS)
    raise ValueError(f"Unsupported gates.profile: {gate_profile!r}")


def _canonical_manifest(
    *,
    profile_id: str,
    inputs: Mapping[str, Any],
    baseline_variant_id: str,
    candidate_variant_ids: tuple[str, ...],
    splits: SplitConfig,
    gates: GateConfig,
    bootstrap: BootstrapConfig,
) -> Dict[str, Any]:
    return {
        "profile": profile_id,
        "inputs": {key: str(value) if isinstance(value, Path) else value for key, value in inputs.items()},
        "baseline_variant_id": baseline_variant_id,
        "candidate_variant_ids": list(candidate_variant_ids),
        "splits": {
            "mode": splits.mode,
            "field": splits.field,
            "holdout_cutoff_utc": splits.holdout_cutoff_utc,
            "holdout_pct": splits.holdout_pct,
        },
        "gates": {
            "profile": gates.profile_id,
            "thresholds": gates.thresholds,
        },
        "bootstrap": {
            "enabled": bootstrap.enabled,
            "n_samples": bootstrap.n_samples,
            "seed": bootstrap.seed,
        },
    }


def _normalize_inputs(base_dir: Path, inputs: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in inputs.items():
        if key.endswith("_path"):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Manifest input {key!r} must be a non-empty path string.")
            path = resolve_path(base_dir, value)
            require_existing_path(path, f"Manifest input {key!r}")
            normalized[key] = path
        else:
            normalized[key] = normalize_json_value(value)
    return normalized


def _load_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Structured file must contain a mapping: {str(path)!r}")
    return payload


def _require_mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest field {key!r} must be a mapping.")
    return value


def _optional_mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest field {key!r} must be a mapping when provided.")
    return value


def _require_string(container: Mapping[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string.")
    return value


def _optional_string(container: Mapping[str, Any], key: str) -> Optional[str]:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string when provided.")
    return value


def _optional_float(container: Mapping[str, Any], key: str) -> Optional[float]:
    value = container.get(key)
    if value is None:
        return None
    return float(value)


def _require_string_list(container: Mapping[str, Any], key: str) -> List[str]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Manifest field {key!r} must be a list.")
    output = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Manifest field {key!r} must contain non-empty strings.")
        output.append(item)
    return output


def _normalize_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Field {field_name!r} must be a mapping when provided.")
    return {str(key): normalize_json_value(item) for key, item in value.items()}
