from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


def reject_unknown_keys(mapping: Mapping[str, Any], allowed_keys: Iterable[str], label: str) -> None:
    allowed = set(allowed_keys)
    unknown = sorted(str(key) for key in mapping.keys() if key not in allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {', '.join(unknown)}")


def reject_duplicate_strings(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{label} contains duplicates: {', '.join(sorted(duplicates))}")


def require_existing_path(path: Path, label: str) -> None:
    if not Path(path).exists():
        raise ValueError(f"{label} does not exist: {path}")


def require_numeric_mapping_values(mapping: Mapping[str, Any], label: str) -> None:
    for key, value in mapping.items():
        if isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be numeric, not boolean.")
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.{key} must be numeric.") from exc


def optional_bool(value: Any, label: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"{label} must be a boolean.")
