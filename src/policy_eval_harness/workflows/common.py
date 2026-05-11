from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


def load_mapping(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping in {path}, found {type(data).__name__}.")
    return data


def require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest field {key!r} must be a mapping.")
    return value


def require_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string.")
    return value


def optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string when provided.")
    return value


def require_string_list(mapping: Mapping[str, Any], key: str) -> List[str]:
    value = mapping.get(key, [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Manifest field {key!r} must be a sequence of strings.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Manifest field {key!r} must contain non-empty strings.")
        result.append(item)
    return result


def require_columns(frame: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Unable to coerce value to bool: {value!r}")
