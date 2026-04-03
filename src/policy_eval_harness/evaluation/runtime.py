from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from policy_eval_harness.replay import canonical_json
from policy_eval_harness.replay.runtime import normalize_json_value

from policy_eval_harness.evaluation.types import (
    BootstrapConfig,
    EvaluationArtifacts,
    EvaluationManifest,
    GateConfig,
    SplitConfig,
)

COMPARISON_PANEL_FILENAME = "comparison_panel.parquet"
SCORECARD_FILENAME = "scorecard.csv"
PROMOTION_DECISIONS_FILENAME = "promotion_decisions.json"

PROFILE_REPLAY = "replay_outcomes_v1"
PROFILE_SELECTION = "selection_panel_v1"
SEQUENTIAL_GATES = "sequential_promotion_v1"
SELECTION_GATES = "selection_promotion_v1"
SPLIT_HOLDOUT = "holdout"
SPLIT_DEV = "dev"

REPLAY_METRICS = (
    "mean_utility",
    "median_utility",
    "paired_coverage_rate",
    "terminal_rate",
    "pathology_rate",
    "error_case_rate",
    "mean_decision_count",
)
SELECTION_METRICS = (
    "balanced_accuracy",
    "tpr",
    "tnr",
    "mean_utility",
    "accept_rate",
    "mean_utility_selected",
    "parse_fail_rate",
    "invalid_output_rate",
    "error_decision_rate",
)

DEFAULT_SEQUENTIAL_THRESHOLDS = {
    "min_delta_mean_utility": 0.0,
    "min_paired_coverage_rate": 1.0,
    "max_candidate_pathology_rate": 1.0,
    "max_candidate_error_case_rate": 1.0,
}

DEFAULT_SELECTION_THRESHOLDS = {
    "min_delta_balanced_accuracy_vs_chance": 0.0,
    "min_delta_mean_utility_vs_accept_all": 0.0,
    "min_delta_mean_utility_vs_random_rate_matched": 0.0,
    "min_accept_rate": 0.0,
    "max_accept_rate": 1.0,
    "max_parse_fail_rate": 1.0,
    "max_invalid_output_rate": 1.0,
    "max_error_decision_rate": 1.0,
}


def run_evaluation_from_manifest(manifest_path: Path, out_dir: Path) -> EvaluationArtifacts:
    manifest = load_evaluation_manifest(Path(manifest_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if manifest.profile_id == PROFILE_REPLAY:
        panel = _build_replay_panel(manifest)
        scorecard, decisions = _evaluate_replay_profile(manifest, panel)
    elif manifest.profile_id == PROFILE_SELECTION:
        panel = _build_selection_panel(manifest)
        scorecard, decisions = _evaluate_selection_profile(manifest, panel)
    else:
        raise ValueError(f"Unsupported evaluation profile: {manifest.profile_id!r}")

    panel = panel.sort_values(["split", "variant_id", "case_id"]).reset_index(drop=True)
    scorecard = scorecard.sort_values(["split", "candidate_variant_id", "metric"]).reset_index(drop=True)

    comparison_panel_path = out_dir / COMPARISON_PANEL_FILENAME
    scorecard_path = out_dir / SCORECARD_FILENAME
    promotion_decisions_path = out_dir / PROMOTION_DECISIONS_FILENAME

    panel.to_parquet(comparison_panel_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    promotion_decisions_path.write_text(canonical_json(decisions) + "\n", encoding="utf-8")

    return EvaluationArtifacts(
        comparison_panel_path=comparison_panel_path,
        scorecard_path=scorecard_path,
        promotion_decisions_path=promotion_decisions_path,
    )


def load_evaluation_manifest(manifest_path: Path) -> EvaluationManifest:
    manifest_path = manifest_path.resolve()
    raw_manifest = _load_mapping(manifest_path)
    base_dir = manifest_path.parent

    profile_id = _require_string(raw_manifest, "profile")
    inputs_raw = _require_mapping(raw_manifest, "inputs")
    baseline_variant_id = _require_string(raw_manifest, "baseline_variant_id")
    candidate_variant_ids = tuple(_require_string_list(raw_manifest, "candidate_variant_ids"))
    if not candidate_variant_ids:
        raise ValueError("Manifest field 'candidate_variant_ids' must not be empty.")

    splits_raw = raw_manifest.get("splits", {})
    if splits_raw is None:
        splits_raw = {}
    if not isinstance(splits_raw, Mapping):
        raise ValueError("Manifest field 'splits' must be a mapping when provided.")
    gates_raw = raw_manifest.get("gates", {})
    if gates_raw is None:
        gates_raw = {}
    if not isinstance(gates_raw, Mapping):
        raise ValueError("Manifest field 'gates' must be a mapping when provided.")
    bootstrap_raw = raw_manifest.get("bootstrap", {})
    if bootstrap_raw is None:
        bootstrap_raw = {}
    if not isinstance(bootstrap_raw, Mapping):
        raise ValueError("Manifest field 'bootstrap' must be a mapping when provided.")

    inputs = _normalize_inputs(base_dir, inputs_raw)
    splits = SplitConfig(
        mode=_optional_string(splits_raw, "mode") or "existing",
        field=_optional_string(splits_raw, "field"),
        holdout_cutoff_utc=_optional_string(splits_raw, "holdout_cutoff_utc"),
        holdout_pct=_optional_float(splits_raw, "holdout_pct"),
    )
    default_gate_profile = SEQUENTIAL_GATES if profile_id == PROFILE_REPLAY else SELECTION_GATES
    gates = GateConfig(
        profile_id=_optional_string(gates_raw, "profile") or default_gate_profile,
        thresholds=_normalize_mapping(gates_raw.get("thresholds", {}), "gates.thresholds"),
    )
    bootstrap = BootstrapConfig(
        enabled=bool(bootstrap_raw.get("enabled", False)),
        n_samples=int(bootstrap_raw.get("n_samples", 0) or 0),
        seed=int(bootstrap_raw.get("seed", 0) or 0),
    )
    if bootstrap.enabled and bootstrap.n_samples <= 0:
        raise ValueError("Bootstrap requires 'n_samples' > 0 when enabled.")

    canonical_manifest = {
        "profile": profile_id,
        "inputs": {key: str(value) if isinstance(value, Path) else value for key, value in inputs.items()},
        "baseline_variant_id": baseline_variant_id,
        "candidate_variant_ids": list(candidate_variant_ids),
        "splits": {
            "mode": splits.mode,
            "field": splits.field,
            "holdout_cutoff_utc": splits.holdout_cutoff_utc,
            "holdout_pct": splits.holdout_pct,
        },
        "gates": {
            "profile": gates.profile_id,
            "thresholds": gates.thresholds,
        },
        "bootstrap": {
            "enabled": bootstrap.enabled,
            "n_samples": bootstrap.n_samples,
            "seed": bootstrap.seed,
        },
    }
    manifest_hash = hashlib.sha256(canonical_json(canonical_manifest).encode("utf-8")).hexdigest()

    return EvaluationManifest(
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        profile_id=profile_id,
        inputs=inputs,
        baseline_variant_id=baseline_variant_id,
        candidate_variant_ids=candidate_variant_ids,
        splits=splits,
        gates=gates,
        bootstrap=bootstrap,
    )


def _build_replay_panel(manifest: EvaluationManifest) -> pd.DataFrame:
    episode_summary_path = _require_path(manifest.inputs, "episode_summary_path")
    summary = _read_table(episode_summary_path)
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
    _require_columns(summary, required_columns, "episode_summary")

    summary = summary.copy()
    summary["case_id"] = summary["case_id"].map(str)
    summary["variant_id"] = summary["variant_id"].map(str)
    summary["summary_metrics"] = summary["summary_metrics_json"].map(_parse_json_mapping)
    summary["utility"] = summary["summary_metrics"].map(lambda value: _optional_float_from_value(value.get("utility")))
    summary["terminated"] = summary["terminated"].map(_to_bool)
    summary["decision_count"] = summary["decision_count"].fillna(0).astype(int)
    summary["error_count"] = summary["error_count"].fillna(0).astype(int)
    summary["start_ts_utc"] = summary["start_ts_utc"].map(_normalize_timestamp_or_none)
    summary["end_ts_utc"] = summary["end_ts_utc"].map(_normalize_timestamp_or_none)

    cases_path = manifest.inputs.get("cases_path")
    if isinstance(cases_path, Path):
        cases = _read_table(cases_path)
        if "case_id" not in cases.columns:
            raise ValueError("Replay cases input requires a 'case_id' column.")
        cases = cases.copy()
        cases["case_id"] = cases["case_id"].map(str)
        summary = summary.merge(cases, on="case_id", how="left", suffixes=("", "_case"))

    summary = _ensure_split(
        summary,
        split_config=manifest.splits,
        default_field_candidates=("split", "opened_at_utc", "start_ts_utc", "end_ts_utc"),
    )
    summary["pathology"] = summary.apply(_is_replay_pathology, axis=1)

    columns = [
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
    ]
    return summary.loc[:, columns]


def _build_selection_panel(manifest: EvaluationManifest) -> pd.DataFrame:
    panel_path = _require_path(manifest.inputs, "panel_path")
    panel = _read_table(panel_path)
    required_columns = {"case_id", "variant_id", "selected", "label", "utility"}
    _require_columns(panel, required_columns, "selection_panel")

    panel = panel.copy()
    panel["case_id"] = panel["case_id"].map(str)
    panel["variant_id"] = panel["variant_id"].map(str)
    panel["selected"] = panel["selected"].map(_to_bool)
    panel["label"] = panel["label"].map(_to_bool)
    panel["utility"] = panel["utility"].map(_optional_float_from_value)
    for field in ("parse_fail", "invalid_output", "error_decision"):
        if field not in panel.columns:
            panel[field] = False
        panel[field] = panel[field].map(_to_bool)

    panel = _ensure_split(
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


def _evaluate_replay_profile(
    manifest: EvaluationManifest, panel: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    score_rows: List[Dict[str, Any]] = []
    score_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    splits = _ordered_splits(panel["split"].unique().tolist())

    for split in splits:
        split_panel = panel[panel["split"] == split]
        for candidate_variant_id in manifest.candidate_variant_ids:
            baseline_frame, candidate_frame, n_union, n_paired = _paired_frames(
                split_panel,
                manifest.baseline_variant_id,
                candidate_variant_id,
            )
            for metric in REPLAY_METRICS:
                baseline_value, candidate_value, delta = _score_replay_metric(
                    metric,
                    baseline_frame,
                    candidate_frame,
                    n_union,
                    n_paired,
                )
                ci_low, ci_high = _bootstrap_ci(
                    manifest.bootstrap,
                    manifest.profile_id,
                    metric,
                    baseline_frame,
                    candidate_frame,
                )
                row = _build_score_row(
                    manifest.profile_id,
                    split,
                    manifest.baseline_variant_id,
                    candidate_variant_id,
                    metric,
                    _metric_group(metric),
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
            _evaluate_replay_candidate(manifest, candidate_variant_id, score_lookup)
            for candidate_variant_id in manifest.candidate_variant_ids
        ],
    }
    return pd.DataFrame(score_rows), decisions


def _evaluate_selection_profile(
    manifest: EvaluationManifest, panel: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    score_rows: List[Dict[str, Any]] = []
    score_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    splits = _ordered_splits(panel["split"].unique().tolist())

    for split in splits:
        split_panel = panel[panel["split"] == split]
        for candidate_variant_id in manifest.candidate_variant_ids:
            baseline_frame, candidate_frame, n_union, n_paired = _paired_frames(
                split_panel,
                manifest.baseline_variant_id,
                candidate_variant_id,
            )
            for metric in SELECTION_METRICS:
                baseline_value, candidate_value, delta = _score_selection_metric(metric, baseline_frame, candidate_frame)
                ci_low, ci_high = _bootstrap_ci(
                    manifest.bootstrap,
                    manifest.profile_id,
                    metric,
                    baseline_frame,
                    candidate_frame,
                )
                row = _build_score_row(
                    manifest.profile_id,
                    split,
                    manifest.baseline_variant_id,
                    candidate_variant_id,
                    metric,
                    _metric_group(metric),
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
            _evaluate_selection_candidate(manifest, panel, candidate_variant_id, score_lookup)
            for candidate_variant_id in manifest.candidate_variant_ids
        ],
    }
    return pd.DataFrame(score_rows), decisions


def _evaluate_replay_candidate(
    manifest: EvaluationManifest,
    candidate_variant_id: str,
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
) -> Dict[str, Any]:
    thresholds = dict(DEFAULT_SEQUENTIAL_THRESHOLDS)
    thresholds.update(manifest.gates.thresholds)

    observed = {
        "delta_mean_utility": _metric_delta(score_lookup, SPLIT_HOLDOUT, candidate_variant_id, "mean_utility"),
        "paired_coverage_rate": _metric_candidate_value(
            score_lookup,
            SPLIT_HOLDOUT,
            candidate_variant_id,
            "paired_coverage_rate",
        ),
        "candidate_pathology_rate": _metric_candidate_value(
            score_lookup,
            SPLIT_HOLDOUT,
            candidate_variant_id,
            "pathology_rate",
        ),
        "candidate_error_case_rate": _metric_candidate_value(
            score_lookup,
            SPLIT_HOLDOUT,
            candidate_variant_id,
            "error_case_rate",
        ),
        "delta_mean_utility_ci_low": _metric_ci_low(
            score_lookup,
            SPLIT_HOLDOUT,
            candidate_variant_id,
            "mean_utility",
        ),
    }
    if any(math.isnan(value) for key, value in observed.items() if key != "delta_mean_utility_ci_low"):
        return _no_verdict(candidate_variant_id, thresholds, observed, "missing holdout paired utility or diagnostics")

    gate_results = [
        _gate_result("min_delta_mean_utility", observed["delta_mean_utility"], thresholds["min_delta_mean_utility"], ">="),
        _gate_result(
            "min_paired_coverage_rate",
            observed["paired_coverage_rate"],
            thresholds["min_paired_coverage_rate"],
            ">=",
        ),
        _gate_result(
            "max_candidate_pathology_rate",
            observed["candidate_pathology_rate"],
            thresholds["max_candidate_pathology_rate"],
            "<=",
        ),
        _gate_result(
            "max_candidate_error_case_rate",
            observed["candidate_error_case_rate"],
            thresholds["max_candidate_error_case_rate"],
            "<=",
        ),
    ]
    if "min_delta_mean_utility_ci_low" in thresholds:
        gate_results.append(
            _gate_result(
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


def _evaluate_selection_candidate(
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
        return _no_verdict(candidate_variant_id, thresholds, {}, "missing holdout candidate rows")
    baseline_holdout, candidate_holdout, n_union, n_paired = _paired_frames(
        holdout_panel,
        manifest.baseline_variant_id,
        candidate_variant_id,
    )
    if baseline_holdout.empty or n_paired == 0:
        return _no_verdict(
            candidate_variant_id,
            thresholds,
            {"paired_coverage_rate": _safe_ratio(n_paired, n_union)},
            "missing paired holdout comparison",
        )

    chance = _build_chance_frame(candidate_holdout, manifest.manifest_hash)
    accept_all = candidate_holdout.assign(selected=True)
    random_rate_matched = _build_random_rate_matched_frame(
        candidate_holdout,
        manifest.manifest_hash,
        candidate_variant_id,
    )

    candidate_metrics = _selection_metric_map(candidate_holdout)
    chance_metrics = _selection_metric_map(chance)
    accept_all_metrics = _selection_metric_map(accept_all)
    random_rate_matched_metrics = _selection_metric_map(random_rate_matched)

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
        "paired_coverage_rate": _safe_ratio(n_paired, n_union),
    }

    gate_results = [
        _gate_result(
            "min_delta_balanced_accuracy_vs_chance",
            observed["delta_balanced_accuracy_vs_chance"],
            thresholds["min_delta_balanced_accuracy_vs_chance"],
            ">=",
        ),
        _gate_result(
            "min_delta_mean_utility_vs_accept_all",
            observed["delta_mean_utility_vs_accept_all"],
            thresholds["min_delta_mean_utility_vs_accept_all"],
            ">=",
        ),
        _gate_result(
            "min_delta_mean_utility_vs_random_rate_matched",
            observed["delta_mean_utility_vs_random_rate_matched"],
            thresholds["min_delta_mean_utility_vs_random_rate_matched"],
            ">=",
        ),
        _gate_result("min_accept_rate", observed["accept_rate"], thresholds["min_accept_rate"], ">="),
        _gate_result("max_accept_rate", observed["accept_rate"], thresholds["max_accept_rate"], "<="),
        _gate_result("max_parse_fail_rate", observed["parse_fail_rate"], thresholds["max_parse_fail_rate"], "<="),
        _gate_result(
            "max_invalid_output_rate",
            observed["invalid_output_rate"],
            thresholds["max_invalid_output_rate"],
            "<=",
        ),
        _gate_result(
            "max_error_decision_rate",
            observed["error_decision_rate"],
            thresholds["max_error_decision_rate"],
            "<=",
        ),
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
            "baseline_scorecard": _scorecard_metric_subset(score_lookup, SPLIT_HOLDOUT, candidate_variant_id),
        },
    }


def _paired_frames(
    panel: pd.DataFrame,
    baseline_variant_id: str,
    candidate_variant_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    baseline = panel[panel["variant_id"] == baseline_variant_id].copy()
    candidate = panel[panel["variant_id"] == candidate_variant_id].copy()
    union_ids = sorted(set(baseline["case_id"]).union(candidate["case_id"]))
    paired_ids = sorted(set(baseline["case_id"]).intersection(candidate["case_id"]))
    baseline = baseline[baseline["case_id"].isin(paired_ids)].sort_values("case_id").reset_index(drop=True)
    candidate = candidate[candidate["case_id"].isin(paired_ids)].sort_values("case_id").reset_index(drop=True)
    return baseline, candidate, len(union_ids), len(paired_ids)


def _score_replay_metric(
    metric: str,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    n_union: int,
    n_paired: int,
) -> Tuple[float, float, float]:
    if metric == "paired_coverage_rate":
        coverage = _safe_ratio(n_paired, n_union)
        return coverage, coverage, 0.0
    if baseline.empty or candidate.empty:
        return math.nan, math.nan, math.nan

    baseline_values = _replay_metric_map(baseline)
    candidate_values = _replay_metric_map(candidate)
    return _metric_triplet(metric, baseline_values, candidate_values)


def _score_selection_metric(metric: str, baseline: pd.DataFrame, candidate: pd.DataFrame) -> Tuple[float, float, float]:
    if baseline.empty or candidate.empty:
        return math.nan, math.nan, math.nan
    baseline_values = _selection_metric_map(baseline)
    candidate_values = _selection_metric_map(candidate)
    return _metric_triplet(metric, baseline_values, candidate_values)


def _replay_metric_map(frame: pd.DataFrame) -> Dict[str, float]:
    utility = frame["utility"].dropna()
    return {
        "mean_utility": float(utility.mean()) if not utility.empty else math.nan,
        "median_utility": float(utility.median()) if not utility.empty else math.nan,
        "terminal_rate": float(frame["terminated"].mean()),
        "pathology_rate": float(frame["pathology"].mean()),
        "error_case_rate": float((frame["error_count"] > 0).mean()),
        "mean_decision_count": float(frame["decision_count"].mean()),
    }


def _selection_metric_map(frame: pd.DataFrame) -> Dict[str, float]:
    tp = int(((frame["selected"]) & (frame["label"])).sum())
    tn = int((~frame["selected"] & ~frame["label"]).sum())
    fp = int(((frame["selected"]) & (~frame["label"])).sum())
    fn = int(((~frame["selected"]) & (frame["label"])).sum())
    tpr = _safe_ratio(tp, tp + fn)
    tnr = _safe_ratio(tn, tn + fp)
    realized_utility = np.where(frame["selected"], frame["utility"].fillna(0.0), 0.0)
    selected_utilities = frame.loc[frame["selected"], "utility"].dropna()
    return {
        "balanced_accuracy": _nanmean([tpr, tnr]),
        "tpr": tpr,
        "tnr": tnr,
        "mean_utility": float(np.mean(realized_utility)) if len(realized_utility) else math.nan,
        "accept_rate": float(frame["selected"].mean()) if len(frame) else math.nan,
        "mean_utility_selected": float(selected_utilities.mean()) if not selected_utilities.empty else math.nan,
        "parse_fail_rate": float(frame["parse_fail"].mean()) if len(frame) else math.nan,
        "invalid_output_rate": float(frame["invalid_output"].mean()) if len(frame) else math.nan,
        "error_decision_rate": float(frame["error_decision"].mean()) if len(frame) else math.nan,
    }


def _metric_triplet(metric: str, baseline_values: Mapping[str, float], candidate_values: Mapping[str, float]) -> Tuple[float, float, float]:
    baseline_value = float(baseline_values.get(metric, math.nan))
    candidate_value = float(candidate_values.get(metric, math.nan))
    if math.isnan(baseline_value) or math.isnan(candidate_value):
        return baseline_value, candidate_value, math.nan
    return baseline_value, candidate_value, candidate_value - baseline_value


def _bootstrap_ci(
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
    rng = np.random.default_rng(_stable_int(f"{bootstrap.seed}:{profile_id}:{metric}"))
    deltas = []
    for _ in range(bootstrap.n_samples):
        sampled_ids = rng.choice(case_ids, size=len(case_ids), replace=True).tolist()
        sample_baseline = baseline_by_case.loc[sampled_ids].reset_index()
        sample_candidate = candidate_by_case.loc[sampled_ids].reset_index()
        if profile_id == PROFILE_REPLAY:
            _, _, delta = _score_replay_metric(metric, sample_baseline, sample_candidate, len(sampled_ids), len(sampled_ids))
        else:
            _, _, delta = _score_selection_metric(metric, sample_baseline, sample_candidate)
        if not math.isnan(delta):
            deltas.append(delta)
    if not deltas:
        return math.nan, math.nan
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _build_score_row(
    profile_id: str,
    split: str,
    baseline_variant_id: str,
    candidate_variant_id: str,
    metric: str,
    metric_group: str,
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
        "metric_group": metric_group,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_cases": int(n_cases),
        "n_paired": int(n_paired),
        "paired_coverage_rate": _safe_ratio(n_paired, n_cases),
    }


def _metric_group(metric: str) -> str:
    if metric in {"mean_utility", "median_utility", "mean_utility_selected"}:
        return "utility"
    if metric in {"terminal_rate", "pathology_rate", "error_case_rate", "parse_fail_rate", "invalid_output_rate", "error_decision_rate"}:
        return "reliability"
    if metric in {"balanced_accuracy", "tpr", "tnr", "accept_rate"}:
        return "behavior"
    return "diagnostics"


def _no_verdict(
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


def _gate_result(name: str, observed: float, threshold: float, operator: str) -> Dict[str, Any]:
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


def _scorecard_metric_subset(
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


def _metric_delta(
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    split: str,
    candidate_variant_id: str,
    metric: str,
) -> float:
    row = score_lookup.get((split, candidate_variant_id, metric))
    if row is None:
        return math.nan
    return float(row["delta"])


def _metric_candidate_value(
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    split: str,
    candidate_variant_id: str,
    metric: str,
) -> float:
    row = score_lookup.get((split, candidate_variant_id, metric))
    if row is None:
        return math.nan
    return float(row["candidate_value"])


def _metric_ci_low(
    score_lookup: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    split: str,
    candidate_variant_id: str,
    metric: str,
) -> float:
    row = score_lookup.get((split, candidate_variant_id, metric))
    if row is None:
        return math.nan
    return float(row["ci_low"])


def _build_chance_frame(frame: pd.DataFrame, manifest_hash: str) -> pd.DataFrame:
    output = frame.copy()
    output["selected"] = output["case_id"].map(lambda case_id: _stable_hash(f"{manifest_hash}:chance:{case_id}") < 0.5)
    return output


def _build_random_rate_matched_frame(frame: pd.DataFrame, manifest_hash: str, candidate_variant_id: str) -> pd.DataFrame:
    output = frame.copy()
    accept_count = int(frame["selected"].sum())
    order = sorted(
        output["case_id"].tolist(),
        key=lambda case_id: _stable_hash(f"{manifest_hash}:rate-matched:{candidate_variant_id}:{case_id}"),
    )
    selected_ids = set(order[:accept_count])
    output["selected"] = output["case_id"].isin(selected_ids)
    return output


def _ensure_split(
    frame: pd.DataFrame,
    split_config: SplitConfig,
    default_field_candidates: Sequence[str],
) -> pd.DataFrame:
    if "split" in frame.columns:
        output = frame.copy()
        output["split"] = output["split"].map(_normalize_split)
        return output

    if split_config.mode != "time_holdout":
        raise ValueError("Input rows must include 'split' unless splits.mode is 'time_holdout'.")

    field = split_config.field
    if field is None:
        for candidate in default_field_candidates:
            if candidate in frame.columns and candidate != "split":
                field = candidate
                break
    if field is None:
        raise ValueError("Unable to infer split field for time_holdout mode.")

    case_frame = frame.loc[:, ["case_id", field]].drop_duplicates(subset=["case_id"]).copy()
    case_frame[field] = case_frame[field].map(_normalize_timestamp_or_none)
    if case_frame[field].isnull().any():
        raise ValueError(f"Split field {field!r} contains null timestamps.")
    case_frame = case_frame.sort_values([field, "case_id"]).reset_index(drop=True)

    if split_config.holdout_cutoff_utc is not None:
        cutoff = _normalize_timestamp_or_none(split_config.holdout_cutoff_utc)
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

    return frame.merge(case_frame[["case_id", "split"]], on="case_id", how="left")


def _ordered_splits(values: Sequence[str]) -> List[str]:
    unique = sorted({str(value) for value in values})
    ordered = []
    for preferred in (SPLIT_DEV, SPLIT_HOLDOUT):
        if preferred in unique:
            ordered.append(preferred)
            unique.remove(preferred)
    ordered.extend(unique)
    return ordered


def _is_replay_pathology(row: pd.Series) -> bool:
    final_status = str(row["final_status"])
    termination_reason = str(row["termination_reason"]) if row["termination_reason"] is not None else ""
    return final_status in {"error", "no_steps"} or termination_reason in {
        "policy_exception",
        "executor_exception",
        "max_steps",
    }


def _normalize_split(value: Any) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("Split values must be non-empty.")
    return normalized


def _parse_json_mapping(value: Any) -> Dict[str, Any]:
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


def _normalize_inputs(base_dir: Path, inputs: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in inputs.items():
        if key.endswith("_path"):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Manifest input {key!r} must be a non-empty path string.")
            normalized[key] = _resolve_path(base_dir, value)
        else:
            normalized[key] = normalize_json_value(value)
    return normalized


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path.suffix!r}")


def _require_columns(frame: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Structured file must contain a mapping: {str(path)!r}")
    return payload


def _require_mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest field {key!r} must be a mapping.")
    return value


def _require_string(container: Mapping[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string.")
    return value


def _optional_string(container: Mapping[str, Any], key: str) -> Optional[str]:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string when provided.")
    return value


def _optional_float(container: Mapping[str, Any], key: str) -> Optional[float]:
    value = container.get(key)
    if value is None:
        return None
    return float(value)


def _require_string_list(container: Mapping[str, Any], key: str) -> List[str]:
    value = container.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Manifest field {key!r} must be a list.")
    output = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Manifest field {key!r} must contain non-empty strings.")
        output.append(item)
    return output


def _normalize_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Field {field_name!r} must be a mapping when provided.")
    return {str(key): normalize_json_value(item) for key, item in value.items()}


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _require_path(container: Mapping[str, Any], key: str) -> Path:
    value = container.get(key)
    if not isinstance(value, Path):
        raise ValueError(f"Manifest input {key!r} must be a path.")
    return value


def _normalize_timestamp_or_none(value: Any) -> Optional[str]:
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


def _optional_float_from_value(value: Any) -> float:
    normalized = normalize_json_value(value)
    if normalized is None:
        return math.nan
    return float(normalized)


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _stable_hash(value: str) -> float:
    return _stable_int(value) / float(2**64 - 1)


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _nanmean(values: Sequence[float]) -> float:
    filtered = [value for value in values if not math.isnan(value)]
    if not filtered:
        return math.nan
    return float(sum(filtered) / len(filtered))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return math.nan
    return float(numerator / denominator)
