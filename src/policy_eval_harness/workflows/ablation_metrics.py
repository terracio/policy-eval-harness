from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from policy_eval_harness.workflows.common import require_columns, to_bool
from policy_eval_harness.workflows.types import AblationManifest, AblationMetricConfig


def metric_value_from_input(
    *,
    frame: pd.DataFrame,
    source_kind: str,
    split: str,
    variant_id: str,
    metric_name: str,
) -> float:
    if source_kind == "scorecard":
        return _scorecard_metric_value(
            frame=frame,
            split=split,
            variant_id=variant_id,
            metric_name=metric_name,
        )
    return _panel_metric_value(
        frame=frame,
        split=split,
        variant_id=variant_id,
        metric_name=metric_name,
    )


def bootstrap_interaction_ci(
    *,
    manifest: AblationManifest,
    frame: pd.DataFrame,
    source_kind: str,
    metric: AblationMetricConfig,
) -> tuple[float | None, float | None]:
    if source_kind != "panel":
        return None, None
    if not bool(manifest.bootstrap.get("enabled", False)):
        return None, None

    n_samples = int(manifest.bootstrap.get("n_samples", 0) or 0)
    if n_samples <= 0:
        raise ValueError("Ablation bootstrap requires 'n_samples' > 0 when enabled.")
    seed = int(manifest.bootstrap.get("seed", 0) or 0)

    split_frame = frame[frame["split"] == manifest.split].copy()
    shared_case_ids = _shared_case_ids(split_frame, manifest.variant_map.values())
    if not shared_case_ids:
        raise ValueError("Ablation bootstrap requires paired case coverage across all four variants.")

    rng = np.random.default_rng(seed)
    interactions: List[float] = []
    for _ in range(n_samples):
        sampled_case_ids = [
            str(case_id)
            for case_id in rng.choice(shared_case_ids, size=len(shared_case_ids), replace=True)
        ]
        sampled = _resample_panel_by_case(split_frame, sampled_case_ids)
        values = {
            key: metric_value_from_input(
                frame=sampled,
                source_kind="panel",
                split=manifest.split,
                variant_id=variant_id,
                metric_name=metric.name,
            )
            for key, variant_id in manifest.variant_map.items()
        }
        baseline_value = values["A0_B0"]
        a_effect = values["A1_B0"] - baseline_value
        b_effect = values["A0_B1"] - baseline_value
        combined_effect = values["A1_B1"] - baseline_value
        interactions.append(combined_effect - a_effect - b_effect)

    return float(np.quantile(interactions, 0.025)), float(np.quantile(interactions, 0.975))


def _scorecard_metric_value(
    *,
    frame: pd.DataFrame,
    split: str,
    variant_id: str,
    metric_name: str,
) -> float:
    required = {
        "split",
        "baseline_variant_id",
        "candidate_variant_id",
        "metric",
        "baseline_value",
        "candidate_value",
    }
    require_columns(frame, required, "ablation.scorecard")
    candidate_rows = frame[
        (frame["split"] == split)
        & (frame["candidate_variant_id"] == variant_id)
        & (frame["metric"] == metric_name)
    ]
    if not candidate_rows.empty:
        return _consistent_numeric_value(
            candidate_rows["candidate_value"],
            split=split,
            variant_id=variant_id,
            metric_name=metric_name,
            value_kind="candidate_value",
        )

    baseline_rows = frame[
        (frame["split"] == split)
        & (frame["baseline_variant_id"] == variant_id)
        & (frame["metric"] == metric_name)
    ]
    if not baseline_rows.empty:
        return _consistent_numeric_value(
            baseline_rows["baseline_value"],
            split=split,
            variant_id=variant_id,
            metric_name=metric_name,
            value_kind="baseline_value",
        )

    raise ValueError(f"Missing scorecard row for split={split!r}, variant={variant_id!r}, metric={metric_name!r}")


def _panel_metric_value(
    *,
    frame: pd.DataFrame,
    split: str,
    variant_id: str,
    metric_name: str,
) -> float:
    variant_frame = frame[(frame["split"] == split) & (frame["variant_id"] == variant_id)]
    if variant_frame.empty:
        raise ValueError(f"Missing panel rows for split={split!r}, variant={variant_id!r}")

    if metric_name == "mean_utility":
        return float(variant_frame["utility"].astype(float).mean())
    if metric_name == "pathology_rate":
        if "pathology" not in variant_frame.columns:
            raise ValueError("Panel input missing 'pathology' for pathology_rate.")
        return float(variant_frame["pathology"].map(to_bool).mean())
    if metric_name == "error_case_rate":
        if "error_count" in variant_frame.columns:
            return float((variant_frame["error_count"].fillna(0).astype(float) > 0).mean())
        if "error_decision" in variant_frame.columns:
            return float(variant_frame["error_decision"].map(to_bool).mean())
        raise ValueError("Panel input missing error field for error_case_rate.")
    if metric_name == "mean_decision_count":
        if "decision_count" not in variant_frame.columns:
            raise ValueError("Panel input missing 'decision_count' for mean_decision_count.")
        return float(variant_frame["decision_count"].astype(float).mean())
    if metric_name == "accept_rate":
        if "selected" not in variant_frame.columns:
            raise ValueError("Panel input missing 'selected' for accept_rate.")
        return float(variant_frame["selected"].map(to_bool).mean())
    if metric_name == "balanced_accuracy":
        if not {"selected", "label"}.issubset(variant_frame.columns):
            raise ValueError("Panel input missing 'selected'/'label' for balanced_accuracy.")
        selected = variant_frame["selected"].map(to_bool)
        labels = variant_frame["label"].map(to_bool)
        return float(_balanced_accuracy(selected, labels))
    raise ValueError(f"Unsupported ablation metric: {metric_name!r}")


def _balanced_accuracy(selected: pd.Series, labels: pd.Series) -> float:
    positive_mask = labels.astype(bool)
    negative_mask = ~positive_mask
    tpr = float(selected[positive_mask].mean()) if positive_mask.any() else float("nan")
    tnr = float((~selected[negative_mask]).mean()) if negative_mask.any() else float("nan")
    if np.isnan(tpr) or np.isnan(tnr):
        return float("nan")
    return (tpr + tnr) / 2.0


def _consistent_numeric_value(
    values: pd.Series,
    *,
    split: str,
    variant_id: str,
    metric_name: str,
    value_kind: str,
) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        raise ValueError(
            "Scorecard {kind} missing numeric value for split={split!r}, variant={variant!r}, metric={metric!r}".format(
                kind=value_kind,
                split=split,
                variant=variant_id,
                metric=metric_name,
            )
        )
    unique_values = sorted({float(value) for value in numeric_values.tolist()})
    if len(unique_values) != 1:
        raise ValueError(
            "Scorecard {kind} is inconsistent for split={split!r}, variant={variant!r}, metric={metric!r}: {values!r}".format(
                kind=value_kind,
                split=split,
                variant=variant_id,
                metric=metric_name,
                values=unique_values,
            )
        )
    return unique_values[0]


def _shared_case_ids(frame: pd.DataFrame, variant_ids: Iterable[str]) -> List[str]:
    shared: set[str] | None = None
    for variant_id in variant_ids:
        variant_cases = set(frame[frame["variant_id"] == variant_id]["case_id"].map(str).tolist())
        shared = variant_cases if shared is None else shared & variant_cases
    return sorted(shared or set())


def _resample_panel_by_case(frame: pd.DataFrame, sampled_case_ids: Sequence[str]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for sample_index, case_id in enumerate(sampled_case_ids):
        case_frame = frame[frame["case_id"] == case_id].copy()
        case_frame["case_id"] = case_frame["case_id"].map(str) + f"__boot_{sample_index:04d}"
        parts.append(case_frame)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()
