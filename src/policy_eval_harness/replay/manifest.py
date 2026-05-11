from __future__ import annotations

import hashlib
import importlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from policy_eval_harness._utils.json import canonical_json, normalize_json_value
from policy_eval_harness._utils.paths import resolve_path
from policy_eval_harness.replay.types import (
    ExecutorConfig,
    OutputConfig,
    ReplayManifest,
    UniverseConfig,
    VariantConfig,
)


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
        cases_path=resolve_path(base_dir, _require_string(universe_data, "cases_path")),
        steps_path=resolve_path(base_dir, _require_string(universe_data, "steps_path")),
    )
    executor = ExecutorConfig(
        import_path=_require_string(executor_data, "import_path"),
        params=_normalize_mapping(executor_data.get("params", {}), "executor.params"),
    )
    variants = _load_variants(resolve_path(base_dir, _require_string(variants_data, "path")))

    run_variant_ids = tuple(_require_string_list(run_data, "variant_ids"))
    if not run_variant_ids:
        raise ValueError("Manifest field 'run.variant_ids' must not be empty.")
    if len(set(run_variant_ids)) != len(run_variant_ids):
        raise ValueError("Manifest field 'run.variant_ids' contains duplicates.")

    variant_map = {variant.variant_id: variant for variant in variants}
    missing_variant_ids = [variant_id for variant_id in run_variant_ids if variant_id not in variant_map]
    if missing_variant_ids:
        raise ValueError("Variant manifest is missing requested variants: " + ", ".join(sorted(missing_variant_ids)))

    output = OutputConfig(
        episode_summary=_optional_string(output_data, "episode_summary") or "episode_summary.csv",
        step_trace=_optional_string(output_data, "step_trace") or "step_trace.parquet",
        run_metadata=_optional_string(output_data, "run_metadata") or "run_metadata.json",
    )
    manifest_hash = hashlib.sha256(canonical_json(_canonical_manifest(
        executor=executor,
        output=output,
        run_variant_ids=run_variant_ids,
        universe_data=universe_data,
        variants=variants,
    )).encode("utf-8")).hexdigest()

    return ReplayManifest(
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        universe=universe,
        executor=executor,
        variants=variants,
        run_variant_ids=run_variant_ids,
        output=output,
    )


def resolve_callable(import_path: str) -> Any:
    module_name, attribute_path = _split_import_path(import_path)
    module = importlib.import_module(module_name)
    target = module
    for attribute_name in attribute_path.split("."):
        target = getattr(target, attribute_name)
    if not callable(target):
        raise TypeError("Resolved import path is not callable: {!r}".format(import_path))
    return target


def _canonical_manifest(
    *,
    executor: ExecutorConfig,
    output: OutputConfig,
    run_variant_ids: Tuple[str, ...],
    universe_data: Mapping[str, Any],
    variants: Tuple[VariantConfig, ...],
) -> Dict[str, Any]:
    return {
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

