from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
import pandas as pd

from policy_eval_harness.evaluation.common import (
    normalize_timestamp_or_none,
    optional_float_from_value,
    parse_json_mapping,
    require_columns,
    require_path,
    to_bool,
)
from policy_eval_harness.evaluation.constants import SPLIT_DEV, SPLIT_HOLDOUT
from policy_eval_harness.evaluation.types import EvaluationManifest, SplitConfig
from policy_eval_harness.io.tables import read_table


def build_replay_panel(manifest: EvaluationManifest) -> pd.DataFrame:
    episode_summary_path = require_path(manifest.inputs, "episode_summary_path")
    summary = read_table(episode_summary_path)
    required_columns = {
        "case_id",
        "variant_id",
        "start_ts_utc",
        "end_ts_utc",
        "terminated",
        "termination_reason",
        "final_status",
        "decision_count",
        "error_count",
        "summary_metrics_json",
    }
    require_columns(summary, required_columns, "episode_summary")

    summary = summary.copy()
    summary["case_id"] = summary["case_id"].map(str)
    summary["variant_id"] = summary["variant_id"].map(str)
    summary["summary_metrics"] = summary["summary_metrics_json"].map(parse_json_mapping)
    summary["utility"] = summary["summary_metrics"].map(lambda value: optional_float_from_value(value.get("utility")))
    summary["terminated"] = summary["terminated"].map(to_bool)
    summary["decision_count"] = summary["decision_count"].fillna(0).astype(int)
    summary["error_count"] = summary["error_count"].fillna(0).astype(int)
    summary["start_ts_utc"] = summary["start_ts_utc"].map(normalize_timestamp_or_none)
    summary["end_ts_utc"] = summary["end_ts_utc"].map(normalize_timestamp_or_none)

    cases_path = manifest.inputs.get("cases_path")
    if isinstance(cases_path, Path):
        cases = read_table(cases_path)
        if "case_id" not in cases.columns:
            raise ValueError("Replay cases input requires a 'case_id' column.")
        cases = cases.copy()
        cases["case_id"] = cases["case_id"].map(str)
        summary = summary.merge(cases, on="case_id", how="left", suffixes=("", "_case"))

    summary = ensure_split(
        summary,
        split_config=manifest.splits,
        default_field_candidates=("split", "opened_at_utc", "start_ts_utc", "end_ts_utc"),
    )
    summary["pathology"] = summary.apply(is_replay_pathology, axis=1)
    return summary.loc[
        :,
        [
            "case_id",
            "variant_id",
            "split",
            "utility",
            "terminated",
            "termination_reason",
            "final_status",
            "decision_count",
            "error_count",
            "pathology",
        ],
    ]


def build_selection_panel(manifest: EvaluationManifest) -> pd.DataFrame:
    panel_path = require_path(manifest.inputs, "panel_path")
    panel = read_table(panel_path)
    require_columns(panel, {"case_id", "variant_id", "selected", "label", "utility"}, "selection_panel")

    panel = panel.copy()
    panel["case_id"] = panel["case_id"].map(str)
    panel["variant_id"] = panel["variant_id"].map(str)
    panel["selected"] = panel["selected"].map(to_bool)
    panel["label"] = panel["label"].map(to_bool)
    panel["utility"] = panel["utility"].map(optional_float_from_value)
    for field in ("parse_fail", "invalid_output", "error_decision"):
        if field not in panel.columns:
            panel[field] = False
        panel[field] = panel[field].map(to_bool)

    panel = ensure_split(
        panel,
        split_config=manifest.splits,
        default_field_candidates=("split", "opened_at_utc", "start_ts_utc", "ts_utc"),
    )
    return panel.loc[
        :,
        [
            "case_id",
            "variant_id",
            "split",
            "selected",
            "label",
            "utility",
            "parse_fail",
            "invalid_output",
            "error_decision",
        ],
    ]


def paired_frames(
    panel: pd.DataFrame,
    baseline_variant_id: str,
    candidate_variant_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    baseline = panel[panel["variant_id"] == baseline_variant_id].copy()
    candidate = panel[panel["variant_id"] == candidate_variant_id].copy()
    reject_duplicate_case_rows(baseline, baseline_variant_id)
    reject_duplicate_case_rows(candidate, candidate_variant_id)
    union_ids = sorted(set(baseline["case_id"]).union(candidate["case_id"]))
    paired_ids = sorted(set(baseline["case_id"]).intersection(candidate["case_id"]))
    baseline = baseline[baseline["case_id"].isin(paired_ids)].sort_values("case_id").reset_index(drop=True)
    candidate = candidate[candidate["case_id"].isin(paired_ids)].sort_values("case_id").reset_index(drop=True)
    return baseline, candidate, len(union_ids), len(paired_ids)


def reject_duplicate_case_rows(frame: pd.DataFrame, variant_id: str) -> None:
    if frame.empty:
        return
    duplicate_mask = frame.duplicated(subset=["split", "variant_id", "case_id"], keep=False)
    if not duplicate_mask.any():
        return
    duplicates = (
        frame.loc[duplicate_mask, ["split", "variant_id", "case_id"]]
        .drop_duplicates()
        .sort_values(["split", "variant_id", "case_id"])
        .head(5)
        .to_dict(orient="records")
    )
    raise ValueError(
        "Duplicate evaluation rows for variant_id={!r}; expected one row per split, variant_id, and case_id: {}".format(
            variant_id,
            duplicates,
        )
    )


def ensure_split(
    frame: pd.DataFrame,
    split_config: SplitConfig,
    default_field_candidates: Sequence[str],
) -> pd.DataFrame:
    if split_config.mode == "existing":
        if "split" not in frame.columns:
            raise ValueError("Input rows must include 'split' when splits.mode is 'existing'.")
        output = frame.copy()
        output["split"] = output["split"].map(normalize_split)
        return output

    if split_config.mode != "time_holdout":
        raise ValueError(f"Unsupported splits.mode: {split_config.mode!r}")

    field = split_config.field
    if field is None:
        for candidate in default_field_candidates:
            if candidate in frame.columns and candidate != "split":
                field = candidate
                break
    if field is None:
        raise ValueError("Unable to infer split field for time_holdout mode.")

    case_frame = frame.loc[:, ["case_id", field]].drop_duplicates(subset=["case_id"]).copy()
    case_frame[field] = case_frame[field].map(normalize_timestamp_or_none)
    if case_frame[field].isnull().any():
        raise ValueError(f"Split field {field!r} contains null timestamps.")
    case_frame = case_frame.sort_values([field, "case_id"]).reset_index(drop=True)

    if split_config.holdout_cutoff_utc is not None:
        cutoff = normalize_timestamp_or_none(split_config.holdout_cutoff_utc)
        case_frame["split"] = np.where(case_frame[field] >= cutoff, SPLIT_HOLDOUT, SPLIT_DEV)
    elif split_config.holdout_pct is not None:
        pct = float(split_config.holdout_pct)
        if pct < 0 or pct > 1:
            raise ValueError("splits.holdout_pct must be between 0 and 1.")
        holdout_n = int(math.ceil(len(case_frame) * pct)) if pct > 0 else 0
        split_values = [SPLIT_DEV] * len(case_frame)
        for index in range(len(case_frame) - holdout_n, len(case_frame)):
            if index >= 0:
                split_values[index] = SPLIT_HOLDOUT
        case_frame["split"] = split_values
    else:
        raise ValueError("time_holdout requires either holdout_cutoff_utc or holdout_pct.")

    output = frame.drop(columns=["split"], errors="ignore")
    return output.merge(case_frame[["case_id", "split"]], on="case_id", how="left")


def is_replay_pathology(row: pd.Series) -> bool:
    final_status = str(row["final_status"])
    termination_reason = str(row["termination_reason"]) if row["termination_reason"] is not None else ""
    return final_status in {"error", "no_steps"} or termination_reason in {
        "policy_exception",
        "executor_exception",
        "max_steps",
    }


def normalize_split(value: Any) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("Split values must be non-empty.")
    return normalized
