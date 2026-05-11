from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


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
    if isinstance(value, np.datetime64):
        return normalize_timestamp(value)
    if isinstance(value, np.generic):
        return normalize_json_value(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return normalize_timestamp(value)
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


def normalize_timestamp(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
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

