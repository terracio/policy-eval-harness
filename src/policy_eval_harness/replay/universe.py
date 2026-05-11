from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from policy_eval_harness._utils.json import normalize_json_value, normalize_timestamp
from policy_eval_harness.replay.constants import (
    CASE_RESERVED_COLUMNS,
    STEP_RESERVED_COLUMNS,
)
from policy_eval_harness.replay.types import ReplayCase, ReplayStep, ReplayUniverse


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


def _read_table(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported table format: {!r}".format(path.suffix))
    return [
        {str(key): value for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


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
    normalized = normalize_timestamp(value)
    if normalized is None:
        raise ValueError("Field '{}' requires a valid UTC timestamp.".format(field_name))
    return normalized
