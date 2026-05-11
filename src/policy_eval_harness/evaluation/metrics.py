from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from policy_eval_harness.evaluation.common import nanmean, safe_ratio
from policy_eval_harness.evaluation.constants import PROFILE_REPLAY
from policy_eval_harness.evaluation.types import BootstrapConfig
from policy_eval_harness.replay import canonical_json


def score_replay_metric(
    metric: str,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    n_union: int,
    n_paired: int,
) -> Tuple[float, float, float]:
    if metric == "paired_coverage_rate":
        coverage = safe_ratio(n_paired, n_union)
        return coverage, coverage, 0.0
    if baseline.empty or candidate.empty:
        return math.nan, math.nan, math.nan

    baseline_values = replay_metric_map(baseline)
    candidate_values = replay_metric_map(candidate)
    return metric_triplet(metric, baseline_values, candidate_values)


def score_selection_metric(metric: str, baseline: pd.DataFrame, candidate: pd.DataFrame) -> Tuple[float, float, float]:
    if baseline.empty or candidate.empty:
        return math.nan, math.nan, math.nan
    baseline_values = selection_metric_map(baseline)
    candidate_values = selection_metric_map(candidate)
    return metric_triplet(metric, baseline_values, candidate_values)


def replay_metric_map(frame: pd.DataFrame) -> Dict[str, float]:
    utility = frame["utility"].dropna()
    return {
        "mean_utility": float(utility.mean()) if not utility.empty else math.nan,
        "median_utility": float(utility.median()) if not utility.empty else math.nan,
        "terminal_rate": float(frame["terminated"].mean()),
        "pathology_rate": float(frame["pathology"].mean()),
        "error_case_rate": float((frame["error_count"] > 0).mean()),
        "mean_decision_count": float(frame["decision_count"].mean()),
    }


def selection_metric_map(frame: pd.DataFrame) -> Dict[str, float]:
    tp = int(((frame["selected"]) & (frame["label"])).sum())
    tn = int((~frame["selected"] & ~frame["label"]).sum())
    fp = int(((frame["selected"]) & (~frame["label"])).sum())
    fn = int(((~frame["selected"]) & (frame["label"])).sum())
    tpr = safe_ratio(tp, tp + fn)
    tnr = safe_ratio(tn, tn + fp)
    realized_utility = np.where(frame["selected"], frame["utility"].fillna(0.0), 0.0)
    selected_utilities = frame.loc[frame["selected"], "utility"].dropna()
    return {
        "balanced_accuracy": nanmean([tpr, tnr]),
        "tpr": tpr,
        "tnr": tnr,
        "mean_utility": float(np.mean(realized_utility)) if len(realized_utility) else math.nan,
        "accept_rate": float(frame["selected"].mean()) if len(frame) else math.nan,
        "mean_utility_selected": float(selected_utilities.mean()) if not selected_utilities.empty else math.nan,
        "parse_fail_rate": float(frame["parse_fail"].mean()) if len(frame) else math.nan,
        "invalid_output_rate": float(frame["invalid_output"].mean()) if len(frame) else math.nan,
        "error_decision_rate": float(frame["error_decision"].mean()) if len(frame) else math.nan,
    }


def metric_triplet(metric: str, baseline_values: Mapping[str, float], candidate_values: Mapping[str, float]) -> Tuple[float, float, float]:
    baseline_value = float(baseline_values.get(metric, math.nan))
    candidate_value = float(candidate_values.get(metric, math.nan))
    if math.isnan(baseline_value) or math.isnan(candidate_value):
        return baseline_value, candidate_value, math.nan
    return baseline_value, candidate_value, candidate_value - baseline_value


def bootstrap_ci(
    bootstrap: BootstrapConfig,
    profile_id: str,
    metric: str,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> Tuple[float, float]:
    if not bootstrap.enabled or baseline.empty or candidate.empty:
        return math.nan, math.nan

    case_ids = baseline["case_id"].tolist()
    if not case_ids:
        return math.nan, math.nan

    baseline_by_case = baseline.set_index("case_id")
    candidate_by_case = candidate.set_index("case_id")
    rng = np.random.default_rng(stable_int(f"{bootstrap.seed}:{profile_id}:{metric}"))
    deltas = []
    for _ in range(bootstrap.n_samples):
        sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True).tolist()
        sample_baseline = baseline_by_case.loc[sampled_ids].reset_index()
        sample_candidate = candidate_by_case.loc[sampled_ids].reset_index()
        if profile_id == PROFILE_REPLAY:
            _, _, delta = score_replay_metric(metric, sample_baseline, sample_candidate, len(sampled_ids), len(sampled_ids))
        else:
            _, _, delta = score_selection_metric(metric, sample_baseline, sample_candidate)
        if not math.isnan(delta):
            deltas.append(delta)
    if not deltas:
        return math.nan, math.nan
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def build_score_row(
    profile_id: str,
    split: str,
    baseline_variant_id: str,
    candidate_variant_id: str,
    metric: str,
    baseline_value: float,
    candidate_value: float,
    delta: float,
    ci_low: float,
    ci_high: float,
    n_cases: int,
    n_paired: int,
) -> Dict[str, Any]:
    return {
        "profile_id": profile_id,
        "split": split,
        "baseline_variant_id": baseline_variant_id,
        "candidate_variant_id": candidate_variant_id,
        "metric": metric,
        "metric_group": metric_group(metric),
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_cases": int(n_cases),
        "n_paired": int(n_paired),
        "paired_coverage_rate": safe_ratio(n_paired, n_cases),
    }


def metric_group(metric: str) -> str:
    if metric in {"mean_utility", "median_utility", "mean_utility_selected"}:
        return "utility"
    if metric in {"terminal_rate", "pathology_rate", "error_case_rate", "parse_fail_rate", "invalid_output_rate", "error_decision_rate"}:
        return "reliability"
    if metric in {"balanced_accuracy", "tpr", "tnr", "accept_rate"}:
        return "behavior"
    return "diagnostics"


def no_verdict(
    candidate_variant_id: str,
    thresholds: Mapping[str, Any],
    observed: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "candidate_variant_id": candidate_variant_id,
        "verdict": "no_verdict",
        "observed": dict(observed),
        "thresholds": dict(thresholds),
        "gate_results": [],
        "failure_reasons": [reason],
    }


def gate_result(name: str, observed: float, threshold: float, operator: str) -> Dict[str, Any]:
    if math.isnan(observed):
        passed = False
    elif operator == ">=":
        passed = observed >= float(threshold)
    elif operator == "<=":
        passed = observed <= float(threshold)
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    return {
        "name": name,
        "operator": operator,
        "observed": observed,
        "threshold": float(threshold),
        "passed": passed,
    }


def scorecard_metric_subset(
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    split: str,
    candidate_variant_id: str,
) -> Dict[str, Dict[str, Any]]:
    subset = {}
    for key, value in score_lookup.items():
        row_split, row_candidate, metric = key
        if row_split == split and row_candidate == candidate_variant_id:
            subset[metric] = dict(value)
    return subset


def metric_delta(
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    split: str,
    candidate_variant_id: str,
    metric: str,
) -> float:
    row = score_lookup.get((split, candidate_variant_id, metric))
    if row is None:
        return math.nan
    return float(row["delta"])


def metric_candidate_value(
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    split: str,
    candidate_variant_id: str,
    metric: str,
) -> float:
    row = score_lookup.get((split, candidate_variant_id, metric))
    if row is None:
        return math.nan
    return float(row["candidate_value"])


def metric_ci_low(
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    split: str,
    candidate_variant_id: str,
    metric: str,
) -> float:
    row = score_lookup.get((split, candidate_variant_id, metric))
    if row is None:
        return math.nan
    return float(row["ci_low"])


def selection_seed_token(frame: pd.DataFrame, candidate_variant_id: str) -> str:
    return canonical_json(
        {
            "candidate_variant_id": candidate_variant_id,
            "case_ids": sorted(frame["case_id"].map(str).tolist()),
        }
    )


def build_chance_frame(frame: pd.DataFrame, seed_token: str) -> pd.DataFrame:
    output = frame.copy()
    output["selected"] = output["case_id"].map(lambda case_id: stable_hash(f"{seed_token}:chance:{case_id}") < 0.5)
    return output


def build_random_rate_matched_frame(frame: pd.DataFrame, seed_token: str) -> pd.DataFrame:
    output = frame.copy()
    accept_count = int(frame["selected"].sum())
    order = sorted(
        output["case_id"].tolist(),
        key=lambda case_id: stable_hash(f"{seed_token}:rate-matched:{case_id}"),
    )
    selected_ids = set(order[:accept_count])
    output["selected"] = output["case_id"].isin(selected_ids)
    return output


def stable_hash(value: str) -> float:
    return stable_int(value) / float(2**64 - 1)


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")

