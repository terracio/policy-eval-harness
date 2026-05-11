from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from policy_eval_harness.evaluation.common import safe_ratio
from policy_eval_harness.evaluation.constants import (
    DEFAULT_SELECTION_THRESHOLDS,
    SELECTION_METRICS,
    SPLIT_DEV,
    SPLIT_HOLDOUT,
)
from policy_eval_harness.evaluation.metrics import (
    bootstrap_ci,
    build_chance_frame,
    build_random_rate_matched_frame,
    build_score_row,
    gate_result,
    no_verdict,
    score_selection_metric,
    scorecard_metric_subset,
    selection_metric_map,
    selection_seed_token,
)
from policy_eval_harness.evaluation.panels import paired_frames
from policy_eval_harness.evaluation.types import EvaluationManifest


def evaluate_selection_profile(
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
            for metric in SELECTION_METRICS:
                baseline_value, candidate_value, delta = score_selection_metric(metric, baseline_frame, candidate_frame)
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
            evaluate_selection_candidate(manifest, panel, candidate_variant_id, score_lookup)
            for candidate_variant_id in manifest.candidate_variant_ids
        ],
    }
    return pd.DataFrame(score_rows), decisions


def evaluate_selection_candidate(
    manifest: EvaluationManifest,
    panel: pd.DataFrame,
    candidate_variant_id: str,
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
) -> Dict[str, Any]:
    thresholds = dict(DEFAULT_SELECTION_THRESHOLDS)
    thresholds.update(manifest.gates.thresholds)

    holdout_panel = panel[panel["split"] == SPLIT_HOLDOUT]
    raw_candidate_holdout = holdout_panel[holdout_panel["variant_id"] == candidate_variant_id]
    if raw_candidate_holdout.empty:
        return no_verdict(candidate_variant_id, thresholds, {}, "missing holdout candidate rows")
    baseline_holdout, candidate_holdout, n_union, n_paired = paired_frames(
        holdout_panel,
        manifest.baseline_variant_id,
        candidate_variant_id,
    )
    if baseline_holdout.empty or n_paired == 0:
        return no_verdict(
            candidate_variant_id,
            thresholds,
            {"paired_coverage_rate": safe_ratio(n_paired, n_union)},
            "missing paired holdout comparison",
        )

    selection_seed = selection_seed_token(candidate_holdout, candidate_variant_id)
    chance = build_chance_frame(candidate_holdout, selection_seed)
    accept_all = candidate_holdout.assign(selected=True)
    random_rate_matched = build_random_rate_matched_frame(candidate_holdout, selection_seed)

    candidate_metrics = selection_metric_map(candidate_holdout)
    chance_metrics = selection_metric_map(chance)
    accept_all_metrics = selection_metric_map(accept_all)
    random_rate_matched_metrics = selection_metric_map(random_rate_matched)

    observed = {
        "delta_balanced_accuracy_vs_chance": candidate_metrics["balanced_accuracy"] - chance_metrics["balanced_accuracy"],
        "delta_mean_utility_vs_accept_all": candidate_metrics["mean_utility"] - accept_all_metrics["mean_utility"],
        "delta_mean_utility_vs_random_rate_matched": (
            candidate_metrics["mean_utility"] - random_rate_matched_metrics["mean_utility"]
        ),
        "accept_rate": candidate_metrics["accept_rate"],
        "parse_fail_rate": candidate_metrics["parse_fail_rate"],
        "invalid_output_rate": candidate_metrics["invalid_output_rate"],
        "error_decision_rate": candidate_metrics["error_decision_rate"],
        "paired_coverage_rate": safe_ratio(n_paired, n_union),
    }

    gate_results = [
        gate_result("min_delta_balanced_accuracy_vs_chance", observed["delta_balanced_accuracy_vs_chance"], thresholds["min_delta_balanced_accuracy_vs_chance"], ">="),
        gate_result("min_delta_mean_utility_vs_accept_all", observed["delta_mean_utility_vs_accept_all"], thresholds["min_delta_mean_utility_vs_accept_all"], ">="),
        gate_result("min_delta_mean_utility_vs_random_rate_matched", observed["delta_mean_utility_vs_random_rate_matched"], thresholds["min_delta_mean_utility_vs_random_rate_matched"], ">="),
        gate_result("min_paired_coverage_rate", observed["paired_coverage_rate"], thresholds["min_paired_coverage_rate"], ">="),
        gate_result("min_accept_rate", observed["accept_rate"], thresholds["min_accept_rate"], ">="),
        gate_result("max_accept_rate", observed["accept_rate"], thresholds["max_accept_rate"], "<="),
        gate_result("max_parse_fail_rate", observed["parse_fail_rate"], thresholds["max_parse_fail_rate"], "<="),
        gate_result("max_invalid_output_rate", observed["invalid_output_rate"], thresholds["max_invalid_output_rate"], "<="),
        gate_result("max_error_decision_rate", observed["error_decision_rate"], thresholds["max_error_decision_rate"], "<="),
    ]
    verdict = "pass" if all(result["passed"] for result in gate_results) else "fail"
    failure_reasons = [result["name"] for result in gate_results if not result["passed"]]
    return {
        "candidate_variant_id": candidate_variant_id,
        "verdict": verdict,
        "observed": observed,
        "thresholds": thresholds,
        "gate_results": gate_results,
        "failure_reasons": failure_reasons,
        "diagnostics": {
            "candidate_metrics": candidate_metrics,
            "chance_metrics": chance_metrics,
            "accept_all_metrics": accept_all_metrics,
            "random_rate_matched_metrics": random_rate_matched_metrics,
            "baseline_scorecard": scorecard_metric_subset(score_lookup, SPLIT_HOLDOUT, candidate_variant_id),
        },
    }


def _ordered_splits(values: list[str]) -> list[str]:
    unique = sorted({str(value) for value in values})
    ordered = []
    for preferred in (SPLIT_DEV, SPLIT_HOLDOUT):
        if preferred in unique:
            ordered.append(preferred)
            unique.remove(preferred)
    ordered.extend(unique)
    return ordered

