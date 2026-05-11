from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCORECARD_FILENAME = "scorecard.csv"
PROMOTION_DECISIONS_FILENAME = "promotion_decisions.json"
MEAN_UTILITY_METRIC = "mean_utility"
DEGRADING_SLOPE_THRESHOLD = -0.001
IMPROVING_SLOPE_THRESHOLD = 0.001
ALLOWED_VERDICTS = {"pass", "fail", "no_verdict"}


@dataclass(frozen=True)
class IterationPoint:
    iteration_index: int
    run_dir: Path
    candidate_variant_id: str
    mean_utility_delta: float
    verdict: str


@dataclass(frozen=True)
class MetricTrend:
    metric: str
    slope: float
    direction: str
    first_value: float | None
    last_value: float | None
    any_regression: bool


@dataclass(frozen=True)
class PolicyIterationTrendReport:
    candidate_variant_id: str | None
    split: str
    iterations_analyzed: int
    mean_utility_trend: MetricTrend
    promotion_rate: float
    pass_count: int
    fail_count: int
    no_verdict_count: int
    any_regression: bool
    consecutive_decline_streak: int
    iteration_points: list[IterationPoint]


def analyze_policy_iteration_trend(
    run_dirs: Sequence[Path],
    *,
    candidate_variant_id: str | None = None,
    window: int = 10,
    split: str = "holdout",
) -> PolicyIterationTrendReport:
    if window < 1:
        raise ValueError("window must be >= 1.")

    normalized_split = _normalize_split(split)
    selected_run_dirs = [Path(run_dir) for run_dir in run_dirs][-window:]
    decisions_by_run = [
        (run_dir, _load_candidate_decisions(run_dir / PROMOTION_DECISIONS_FILENAME))
        for run_dir in selected_run_dirs
    ]
    resolved_candidate_variant_id = _resolve_candidate_variant_id(decisions_by_run, candidate_variant_id)

    iteration_points: list[IterationPoint] = []
    for iteration_index, (run_dir, candidate_decisions) in enumerate(decisions_by_run, start=1):
        if resolved_candidate_variant_id is None:
            raise ValueError("Unable to resolve candidate_variant_id for trend analysis.")
        candidate_decision = candidate_decisions.get(resolved_candidate_variant_id)
        if candidate_decision is None:
            raise ValueError(
                f"candidate_variant_id {resolved_candidate_variant_id!r} is missing from "
                f"{run_dir / PROMOTION_DECISIONS_FILENAME}."
            )
        verdict = _normalize_verdict(candidate_decision.get("verdict"), run_dir)
        mean_utility_delta = _read_scorecard_delta(
            run_dir / SCORECARD_FILENAME,
            split=normalized_split,
            candidate_variant_id=resolved_candidate_variant_id,
            metric=MEAN_UTILITY_METRIC,
            verdict=verdict,
        )
        iteration_points.append(
            IterationPoint(
                iteration_index=iteration_index,
                run_dir=run_dir,
                candidate_variant_id=resolved_candidate_variant_id,
                mean_utility_delta=mean_utility_delta,
                verdict=verdict,
            )
        )

    return _build_report(
        candidate_variant_id=resolved_candidate_variant_id,
        split=normalized_split,
        iteration_points=iteration_points,
    )


def _build_report(
    *,
    candidate_variant_id: str | None,
    split: str,
    iteration_points: Sequence[IterationPoint],
) -> PolicyIterationTrendReport:
    pass_count = sum(point.verdict == "pass" for point in iteration_points)
    fail_count = sum(point.verdict == "fail" for point in iteration_points)
    no_verdict_count = sum(point.verdict == "no_verdict" for point in iteration_points)
    iterations_analyzed = len(iteration_points)
    promotion_rate = pass_count / iterations_analyzed if iterations_analyzed else 0.0
    values = [point.mean_utility_delta for point in iteration_points]
    finite_values = [value for value in values if math.isfinite(value)]
    first_value = finite_values[0] if finite_values else None
    last_value = finite_values[-1] if finite_values else None

    if len(finite_values) < 2:
        mean_utility_trend = MetricTrend(
            metric=MEAN_UTILITY_METRIC,
            slope=0.0,
            direction="stable",
            first_value=first_value,
            last_value=last_value,
            any_regression=False,
        )
        return PolicyIterationTrendReport(
            candidate_variant_id=candidate_variant_id,
            split=split,
            iterations_analyzed=iterations_analyzed,
            mean_utility_trend=mean_utility_trend,
            promotion_rate=promotion_rate,
            pass_count=pass_count,
            fail_count=fail_count,
            no_verdict_count=no_verdict_count,
            any_regression=False,
            consecutive_decline_streak=_consecutive_decline_streak(values),
            iteration_points=list(iteration_points),
        )

    regression = statistics.linear_regression(range(1, len(finite_values) + 1), finite_values)
    slope = float(regression.slope)
    direction = _trend_direction(slope)
    any_regression = slope < DEGRADING_SLOPE_THRESHOLD
    mean_utility_trend = MetricTrend(
        metric=MEAN_UTILITY_METRIC,
        slope=slope,
        direction=direction,
        first_value=first_value,
        last_value=last_value,
        any_regression=any_regression,
    )
    return PolicyIterationTrendReport(
        candidate_variant_id=candidate_variant_id,
        split=split,
        iterations_analyzed=iterations_analyzed,
        mean_utility_trend=mean_utility_trend,
        promotion_rate=promotion_rate,
        pass_count=pass_count,
        fail_count=fail_count,
        no_verdict_count=no_verdict_count,
        any_regression=any_regression,
        consecutive_decline_streak=_consecutive_decline_streak(values),
        iteration_points=list(iteration_points),
    )


def _resolve_candidate_variant_id(
    decisions_by_run: Sequence[tuple[Path, Mapping[str, Mapping[str, Any]]]],
    requested_candidate_variant_id: str | None,
) -> str | None:
    if requested_candidate_variant_id is not None:
        return requested_candidate_variant_id
    if not decisions_by_run:
        return None

    detected_candidate_ids: list[str] = []
    for run_dir, candidate_decisions in decisions_by_run:
        candidate_ids = sorted(candidate_decisions)
        if len(candidate_ids) != 1:
            raise ValueError(
                "candidate_variant_id is required when analyzed runs do not each contain exactly one candidate. "
                f"Ambiguous run: {run_dir}."
            )
        detected_candidate_ids.append(candidate_ids[0])

    unique_candidate_ids = set(detected_candidate_ids)
    if len(unique_candidate_ids) != 1:
        raise ValueError(
            "candidate_variant_id is required when auto-detected candidate ids differ across runs."
        )
    return detected_candidate_ids[0]


def _load_candidate_decisions(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{path} must contain a 'candidates' list.")

    decisions: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{path} contains an invalid candidate payload.")
        candidate_id = candidate.get("candidate_variant_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{path} contains a candidate without candidate_variant_id.")
        if candidate_id in decisions:
            raise ValueError(f"{path} contains duplicate candidate_variant_id {candidate_id!r}.")
        decisions[candidate_id] = candidate
    return decisions


def _read_scorecard_delta(
    path: Path,
    *,
    split: str,
    candidate_variant_id: str,
    metric: str,
    verdict: str,
) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}.")

    matches: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (
                _normalize_split(row.get("split", "")) == split
                and row.get("candidate_variant_id") == candidate_variant_id
                and row.get("metric") == metric
            ):
                matches.append(row)

    if not matches:
        raise ValueError(
            f"Missing scorecard row for split={split!r}, metric={metric!r}, "
            f"candidate_variant_id={candidate_variant_id!r} in {path}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple scorecard rows for split={split!r}, metric={metric!r}, "
            f"candidate_variant_id={candidate_variant_id!r} in {path}."
        )

    raw_delta = matches[0].get("delta")
    try:
        if raw_delta is None:
            raise ValueError("missing delta")
        delta = float(raw_delta)
    except (TypeError, ValueError) as exc:
        if verdict == "no_verdict":
            return math.nan
        raise ValueError(
            f"Scorecard delta is invalid for split={split!r}, metric={metric!r}, "
            f"candidate_variant_id={candidate_variant_id!r} in {path}."
        ) from exc
    if not math.isfinite(delta):
        if verdict == "no_verdict":
            return math.nan
        raise ValueError(
            f"Scorecard delta is invalid for split={split!r}, metric={metric!r}, "
            f"candidate_variant_id={candidate_variant_id!r} in {path}."
        )
    return delta


def _normalize_split(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("split must be non-empty.")
    return normalized


def _normalize_verdict(value: Any, run_dir: Path) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{run_dir / PROMOTION_DECISIONS_FILENAME} contains a non-string verdict.")
    verdict = value.strip().lower()
    if verdict not in ALLOWED_VERDICTS:
        raise ValueError(
            f"{run_dir / PROMOTION_DECISIONS_FILENAME} contains unsupported verdict {value!r}."
        )
    return verdict


def _trend_direction(slope: float) -> str:
    if slope < DEGRADING_SLOPE_THRESHOLD:
        return "degrading"
    if slope > IMPROVING_SLOPE_THRESHOLD:
        return "improving"
    return "stable"


def _consecutive_decline_streak(values: Sequence[float]) -> int:
    if not values or not math.isfinite(values[-1]):
        return 0

    streak = 0
    for index in range(len(values) - 1, 0, -1):
        if not math.isfinite(values[index - 1]):
            break
        if values[index] < values[index - 1]:
            streak += 1
        else:
            break
    return streak
