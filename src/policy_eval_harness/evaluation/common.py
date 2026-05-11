from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from policy_eval_harness._utils.json import normalize_json_value


def require_columns(frame: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def require_path(container: Mapping[str, Any], key: str) -> Path:
    value = container.get(key)
    if not isinstance(value, Path):
        raise ValueError(f"Manifest input {key!r} must be a path.")
    return value


def normalize_timestamp_or_none(value: Any) -> Optional[str]:
    normalized = normalize_json_value(value)
    if normalized is None:
        return None
    timestamp = pd.Timestamp(normalized)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    iso = timestamp.isoformat().replace("+00:00", "Z")
    if "." not in iso:
        return iso
    prefix, suffix = iso.split(".", 1)
    fraction = suffix[:-1].rstrip("0")
    return prefix + ("." + fraction if fraction else "") + "Z"


def optional_float_from_value(value: Any) -> float:
    normalized = normalize_json_value(value)
    if normalized is None:
        return math.nan
    return float(normalized)


def to_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    return bool(value)


def parse_json_mapping(value: Any) -> Dict[str, Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, Mapping):
        return {str(key): normalize_json_value(item) for key, item in value.items()}
    if not isinstance(value, str):
        raise ValueError("Expected JSON object string for summary_metrics_json.")
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise ValueError("summary_metrics_json must decode to a mapping.")
    return {str(key): normalize_json_value(item) for key, item in parsed.items()}


def nanmean(values: Sequence[float]) -> float:
    filtered = [value for value in values if not math.isnan(value)]
    if not filtered:
        return math.nan
    return float(sum(filtered) / len(filtered))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return math.nan
    return float(numerator / denominator)
