from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from policy_eval_harness._utils.paths import resolve_path
from policy_eval_harness.evaluation.constants import (
    PROFILE_REPLAY,
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
from policy_eval_harness.replay.runtime import normalize_json_value


def load_evaluation_manifest(manifest_path: Path) -> EvaluationManifest:
    manifest_path = manifest_path.resolve()
    raw_manifest = _load_mapping(manifest_path)
    base_dir = manifest_path.parent

    profile_id = _require_string(raw_manifest, "profile")
    inputs_raw = _require_mapping(raw_manifest, "inputs")
    baseline_variant_id = _require_string(raw_manifest, "baseline_variant_id")
    candidate_variant_ids = tuple(_require_string_list(raw_manifest, "candidate_variant_ids"))
    if not candidate_variant_ids:
        raise ValueError("Manifest field 'candidate_variant_ids' must not be empty.")

    splits_raw = _optional_mapping(raw_manifest, "splits")
    gates_raw = _optional_mapping(raw_manifest, "gates")
    bootstrap_raw = _optional_mapping(raw_manifest, "bootstrap")

    inputs = _normalize_inputs(base_dir, inputs_raw)
    splits = SplitConfig(
        mode=_optional_string(splits_raw, "mode") or "existing",
        field=_optional_string(splits_raw, "field"),
        holdout_cutoff_utc=_optional_string(splits_raw, "holdout_cutoff_utc"),
        holdout_pct=_optional_float(splits_raw, "holdout_pct"),
    )
    default_gate_profile = SEQUENTIAL_GATES if profile_id == PROFILE_REPLAY else SELECTION_GATES
    gates = GateConfig(
        profile_id=_optional_string(gates_raw, "profile") or default_gate_profile,
        thresholds=_normalize_mapping(gates_raw.get("thresholds", {}), "gates.thresholds"),
    )
    bootstrap = BootstrapConfig(
        enabled=bool(bootstrap_raw.get("enabled", False)),
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
            normalized[key] = resolve_path(base_dir, value)
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

