from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from policy_eval_harness.evaluation.constants import (
    DEFAULT_SEQUENTIAL_THRESHOLDS,
    REPLAY_METRICS,
    SPLIT_HOLDOUT,
)
from policy_eval_harness.evaluation.metrics import (
    bootstrap_ci,
    build_score_row,
    gate_result,
    metric_candidate_value,
    metric_ci_low,
    metric_delta,
    no_verdict,
    score_replay_metric,
)
from policy_eval_harness.evaluation.panels import paired_frames
from policy_eval_harness.evaluation.types import EvaluationManifest


def evaluate_replay_profile(
    manifest: EvaluationManifest, panel: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    score_rows: list[Dict[str, Any]] = []
    score_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    splits = _ordered_splits(panel["split"].unique().tolist())

    for split in splits:
        split_panel = panel[panel["split"] == split]
        for candidate_variant_id in manifest.candidate_variant_ids:
            baseline_frame, candidate_frame, n_union, n_paired = paired_frames(
                split_panel,
                manifest.baseline_variant_id,
                candidate_variant_id,
            )
            for metric in REPLAY_METRICS:
                baseline_value, candidate_value, delta = score_replay_metric(
                    metric,
                    baseline_frame,
                    candidate_frame,
                    n_union,
                    n_paired,
                )
                ci_low, ci_high = bootstrap_ci(
                    manifest.bootstrap,
                    manifest.profile_id,
                    metric,
                    baseline_frame,
                    candidate_frame,
                )
                row = build_score_row(
                    manifest.profile_id,
                    split,
                    manifest.baseline_variant_id,
                    candidate_variant_id,
                    metric,
                    baseline_value,
                    candidate_value,
                    delta,
                    ci_low,
                    ci_high,
                    n_union,
                    n_paired,
                )
                score_rows.append(row)
                score_lookup[(split, candidate_variant_id, metric)] = row

    decisions = {
        "profile_id": manifest.profile_id,
        "gate_profile_id": manifest.gates.profile_id,
        "evaluation_split": SPLIT_HOLDOUT,
        "baseline_variant_id": manifest.baseline_variant_id,
        "candidates": [
            evaluate_replay_candidate(manifest, candidate_variant_id, score_lookup)
            for candidate_variant_id in manifest.candidate_variant_ids
        ],
    }
    return pd.DataFrame(score_rows), decisions


def evaluate_replay_candidate(
    manifest: EvaluationManifest,
    candidate_variant_id: str,
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
) -> Dict[str, Any]:
    thresholds = dict(DEFAULT_SEQUENTIAL_THRESHOLDS)
    thresholds.update(manifest.gates.thresholds)

    observed = {
        "delta_mean_utility": metric_delta(score_lookup, SPLIT_HOLDOUT, candidate_variant_id, "mean_utility"),
        "paired_coverage_rate": metric_candidate_value(
            score_lookup,
            SPLIT_HOLDOUT,
            candidate_variant_id,
            "paired_coverage_rate",
        ),
        "candidate_pathology_rate": metric_candidate_value(
            score_lookup,
            SPLIT_HOLDOUT,
            candidate_variant_id,
            "pathology_rate",
        ),
        "candidate_error_case_rate": metric_candidate_value(
            score_lookup,
            SPLIT_HOLDOUT,
            candidate_variant_id,
            "error_case_rate",
        ),
        "delta_mean_utility_ci_low": metric_ci_low(
            score_lookup,
            SPLIT_HOLDOUT,
            candidate_variant_id,
            "mean_utility",
        ),
    }
    if any(math.isnan(value) for key, value in observed.items() if key != "delta_mean_utility_ci_low"):
        return no_verdict(candidate_variant_id, thresholds, observed, "missing holdout paired utility or diagnostics")

    gate_results = [
        gate_result("min_delta_mean_utility", observed["delta_mean_utility"], thresholds["min_delta_mean_utility"], ">="),
        gate_result(
            "min_paired_coverage_rate",
            observed["paired_coverage_rate"],
            thresholds["min_paired_coverage_rate"],
            ">=",
        ),
        gate_result(
            "max_candidate_pathology_rate",
            observed["candidate_pathology_rate"],
            thresholds["max_candidate_pathology_rate"],
            "<=",
        ),
        gate_result(
            "max_candidate_error_case_rate",
            observed["candidate_error_case_rate"],
            thresholds["max_candidate_error_case_rate"],
            "<=",
        ),
    ]
    if "min_delta_mean_utility_ci_low" in thresholds:
        gate_results.append(
            gate_result(
                "min_delta_mean_utility_ci_low",
                observed["delta_mean_utility_ci_low"],
                float(thresholds["min_delta_mean_utility_ci_low"]),
                ">=",
            )
        )

    verdict = "pass" if all(result["passed"] for result in gate_results) else "fail"
    failure_reasons = [result["name"] for result in gate_results if not result["passed"]]
    return {
        "candidate_variant_id": candidate_variant_id,
        "verdict": verdict,
        "observed": observed,
        "thresholds": thresholds,
        "gate_results": gate_results,
        "failure_reasons": failure_reasons,
    }


def _ordered_splits(values: list[str]) -> list[str]:
    unique = sorted({str(value) for value in values})
    ordered = []
    for preferred in ("dev", SPLIT_HOLDOUT):
        if preferred in unique:
            ordered.append(preferred)
            unique.remove(preferred)
    ordered.extend(unique)
    return ordered

