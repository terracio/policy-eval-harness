from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCORECARD_FIELDNAMES = [
    "profile_id",
    "split",
    "baseline_variant_id",
    "candidate_variant_id",
    "metric",
    "metric_group",
    "baseline_value",
    "candidate_value",
    "delta",
    "ci_low",
    "ci_high",
    "n_cases",
    "n_paired",
    "paired_coverage_rate",
]


class EvaluationTrendFixtureMixin:
    root: Path

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_sequence(
        self,
        runs: Sequence[Mapping[str, tuple[float | None, str]]],
        *,
        missing_scorecard_indices: Iterable[int] = (),
        missing_decisions_indices: Iterable[int] = (),
        omit_holdout_mean_utility_indices: Iterable[int] = (),
        blank_delta_indices: Iterable[int] = (),
        invalid_decisions_root_indices: Iterable[int] = (),
    ) -> list[Path]:
        missing_scorecard = set(missing_scorecard_indices)
        missing_decisions = set(missing_decisions_indices)
        omit_holdout_mean_utility = set(omit_holdout_mean_utility_indices)
        blank_delta = set(blank_delta_indices)
        invalid_decisions_root = set(invalid_decisions_root_indices)

        run_dirs: list[Path] = []
        for index, candidate_map in enumerate(runs, start=1):
            run_dir = self.root / f"run-{index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if index - 1 not in missing_scorecard:
                self._write_scorecard(
                    run_dir / "scorecard.csv",
                    candidate_map,
                    omit_holdout_mean_utility=index - 1 in omit_holdout_mean_utility,
                    blank_delta=index - 1 in blank_delta,
                )
            if index - 1 not in missing_decisions:
                self._write_promotion_decisions(
                    run_dir / "promotion_decisions.json",
                    candidate_map,
                    invalid_root=index - 1 in invalid_decisions_root,
                )
            run_dirs.append(run_dir)
        return run_dirs

    def _write_scorecard(
        self,
        path: Path,
        candidates: Mapping[str, tuple[float | None, str]],
        *,
        omit_holdout_mean_utility: bool = False,
        blank_delta: bool = False,
    ) -> None:
        rows = []
        for candidate_variant_id, (delta, _) in candidates.items():
            if not omit_holdout_mean_utility:
                rows.append(
                    self._scorecard_row(
                        split="holdout",
                        candidate_variant_id=candidate_variant_id,
                        metric="mean_utility",
                        metric_group="utility",
                        delta=None if blank_delta else delta,
                    )
                )
            rows.append(
                self._scorecard_row(
                    split="dev",
                    candidate_variant_id=candidate_variant_id,
                    metric="mean_utility",
                    metric_group="utility",
                    delta=1.0 if delta is None else delta + 1.0,
                )
            )
            rows.append(
                self._scorecard_row(
                    split="holdout",
                    candidate_variant_id=candidate_variant_id,
                    metric="median_utility",
                    metric_group="utility",
                    delta=0.0 if delta is None else delta,
                )
            )

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCORECARD_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def _write_promotion_decisions(
        self,
        path: Path,
        candidates: Mapping[str, tuple[float | None, str]],
        *,
        invalid_root: bool = False,
    ) -> None:
        if invalid_root:
            path.write_text("[]\n", encoding="utf-8")
            return
        payload = {
            "profile_id": "replay_outcomes_v1",
            "gate_profile_id": "sequential_promotion_v1",
            "evaluation_split": "holdout",
            "baseline_variant_id": "baseline",
            "candidates": [
                {
                    "candidate_variant_id": candidate_variant_id,
                    "verdict": verdict,
                    "failure_reasons": [] if verdict == "pass" else ["trend-test"],
                    "gate_results": [],
                    "observed": {"delta_mean_utility": delta},
                    "thresholds": {"min_delta_mean_utility": 0.0},
                }
                for candidate_variant_id, (delta, verdict) in candidates.items()
            ],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _scorecard_row(
        self,
        *,
        split: str,
        candidate_variant_id: str,
        metric: str,
        metric_group: str,
        delta: float | None,
    ) -> dict[str, object]:
        baseline_value = 0.8
        candidate_value = "" if delta is None else baseline_value + delta
        return {
            "profile_id": "replay_outcomes_v1",
            "split": split,
            "baseline_variant_id": "baseline",
            "candidate_variant_id": candidate_variant_id,
            "metric": metric,
            "metric_group": metric_group,
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
            "delta": "" if delta is None else delta,
            "ci_low": "",
            "ci_high": "",
            "n_cases": 8,
            "n_paired": 8,
            "paired_coverage_rate": 1.0,
        }
