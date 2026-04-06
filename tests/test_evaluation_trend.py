from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from typer.testing import CliRunner

from policy_eval_harness.cli import app
from policy_eval_harness.evaluation import analyze_policy_iteration_trend

CLI_RUNNER = CliRunner()

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


class EvaluationTrendTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_degrading_sequence_detects_regression_before_gate_failure(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (0.14, "pass")},
                {"candidate": (0.09, "pass")},
                {"candidate": (0.05, "pass")},
                {"candidate": (0.01, "pass")},
                {"candidate": (-0.03, "fail")},
            ]
        )

        report = analyze_policy_iteration_trend(run_dirs)

        self.assertEqual(report.candidate_variant_id, "candidate")
        self.assertEqual(report.split, "holdout")
        self.assertEqual(report.iterations_analyzed, 5)
        self.assertEqual(report.mean_utility_trend.metric, "mean_utility")
        self.assertEqual(report.mean_utility_trend.direction, "degrading")
        self.assertTrue(report.mean_utility_trend.any_regression)
        self.assertAlmostEqual(report.mean_utility_trend.first_value, 0.14)
        self.assertAlmostEqual(report.mean_utility_trend.last_value, -0.03)
        self.assertTrue(report.any_regression)
        self.assertAlmostEqual(report.promotion_rate, 0.8)
        self.assertEqual(report.pass_count, 4)
        self.assertEqual(report.fail_count, 1)
        self.assertEqual(report.no_verdict_count, 0)
        self.assertEqual(report.consecutive_decline_streak, 4)
        self.assertEqual([point.verdict for point in report.iteration_points], ["pass", "pass", "pass", "pass", "fail"])

    def test_improving_sequence_reports_positive_slope(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (-0.08, "fail")},
                {"candidate": (-0.02, "pass")},
                {"candidate": (0.02, "pass")},
                {"candidate": (0.07, "pass")},
            ]
        )

        report = analyze_policy_iteration_trend(run_dirs)

        self.assertEqual(report.mean_utility_trend.direction, "improving")
        self.assertGreater(report.mean_utility_trend.slope, 0.001)
        self.assertFalse(report.any_regression)
        self.assertEqual(report.consecutive_decline_streak, 0)

    def test_stable_sequence_reports_no_regression(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (0.05, "pass")},
                {"candidate": (0.05, "pass")},
                {"candidate": (0.05, "pass")},
            ]
        )

        report = analyze_policy_iteration_trend(run_dirs)

        self.assertEqual(report.mean_utility_trend.direction, "stable")
        self.assertAlmostEqual(report.mean_utility_trend.slope, 0.0)
        self.assertFalse(report.any_regression)
        self.assertEqual(report.consecutive_decline_streak, 0)

    def test_mixed_verdicts_are_counted(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (0.04, "pass")},
                {"candidate": (0.03, "no_verdict")},
                {"candidate": (0.02, "fail")},
                {"candidate": (0.01, "pass")},
            ]
        )

        report = analyze_policy_iteration_trend(run_dirs)

        self.assertEqual(report.pass_count, 2)
        self.assertEqual(report.fail_count, 1)
        self.assertEqual(report.no_verdict_count, 1)
        self.assertAlmostEqual(report.promotion_rate, 0.5)

    def test_explicit_candidate_selection_across_multi_candidate_runs(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate-a": (0.08, "pass"), "candidate-b": (-0.02, "fail")},
                {"candidate-a": (0.09, "pass"), "candidate-b": (-0.01, "fail")},
                {"candidate-a": (0.10, "pass"), "candidate-b": (0.00, "pass")},
            ]
        )

        report = analyze_policy_iteration_trend(run_dirs, candidate_variant_id="candidate-b")

        self.assertEqual(report.candidate_variant_id, "candidate-b")
        self.assertEqual([point.candidate_variant_id for point in report.iteration_points], ["candidate-b"] * 3)
        self.assertEqual(report.mean_utility_trend.direction, "improving")

    def test_auto_detects_candidate_when_each_run_has_one_consistent_candidate(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (0.02, "pass")},
                {"candidate": (0.01, "pass")},
            ]
        )

        report = analyze_policy_iteration_trend(run_dirs)

        self.assertEqual(report.candidate_variant_id, "candidate")

    def test_rejects_ambiguous_candidate_auto_detection_for_multi_candidate_runs(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate-a": (0.03, "pass"), "candidate-b": (0.02, "pass")},
                {"candidate-a": (0.02, "pass"), "candidate-b": (0.01, "pass")},
            ]
        )

        with self.assertRaisesRegex(ValueError, "candidate_variant_id"):
            analyze_policy_iteration_trend(run_dirs)

    def test_rejects_inconsistent_single_candidate_auto_detection(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate-a": (0.03, "pass")},
                {"candidate-b": (0.02, "pass")},
            ]
        )

        with self.assertRaisesRegex(ValueError, "candidate_variant_id"):
            analyze_policy_iteration_trend(run_dirs)

    def test_errors_when_required_scorecard_is_missing(self) -> None:
        run_dirs = self._write_sequence([{"candidate": (0.01, "pass")}], missing_scorecard_indices={0})

        with self.assertRaisesRegex(FileNotFoundError, "scorecard.csv"):
            analyze_policy_iteration_trend(run_dirs)

    def test_errors_when_required_decisions_file_is_missing(self) -> None:
        run_dirs = self._write_sequence([{"candidate": (0.01, "pass")}], missing_decisions_indices={0})

        with self.assertRaisesRegex(FileNotFoundError, "promotion_decisions.json"):
            analyze_policy_iteration_trend(run_dirs)

    def test_errors_when_scorecard_row_is_missing_for_requested_candidate(self) -> None:
        run_dirs = self._write_sequence(
            [{"candidate": (0.03, "pass")}],
            omit_holdout_mean_utility_indices={0},
        )

        with self.assertRaisesRegex(ValueError, "mean_utility"):
            analyze_policy_iteration_trend(run_dirs)

    def test_errors_when_requested_candidate_is_missing_from_a_run(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate-a": (0.03, "pass"), "candidate-b": (0.02, "pass")},
                {"candidate-a": (0.01, "pass")},
            ]
        )

        with self.assertRaisesRegex(ValueError, "candidate-b"):
            analyze_policy_iteration_trend(run_dirs, candidate_variant_id="candidate-b")

    def test_window_uses_only_last_run_directories(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (0.10, "pass")},
                {"candidate": (0.05, "pass")},
                {"candidate": (0.02, "pass")},
                {"candidate": (0.02, "pass")},
            ]
        )

        report = analyze_policy_iteration_trend(run_dirs, window=2)

        self.assertEqual(report.iterations_analyzed, 2)
        self.assertEqual(report.mean_utility_trend.direction, "stable")
        self.assertAlmostEqual(report.mean_utility_trend.first_value, 0.02)
        self.assertAlmostEqual(report.mean_utility_trend.last_value, 0.02)

    def test_returns_stable_report_for_single_iteration(self) -> None:
        run_dirs = self._write_sequence([{"candidate": (0.05, "pass")}])

        report = analyze_policy_iteration_trend(run_dirs)

        self.assertEqual(report.iterations_analyzed, 1)
        self.assertEqual(report.mean_utility_trend.direction, "stable")
        self.assertAlmostEqual(report.mean_utility_trend.slope, 0.0)
        self.assertFalse(report.any_regression)

    def test_returns_stable_report_for_empty_input(self) -> None:
        report = analyze_policy_iteration_trend([])

        self.assertEqual(report.iterations_analyzed, 0)
        self.assertEqual(report.mean_utility_trend.direction, "stable")
        self.assertAlmostEqual(report.promotion_rate, 0.0)
        self.assertEqual(report.iteration_points, [])

    def test_rejects_non_positive_window(self) -> None:
        run_dirs = self._write_sequence([{"candidate": (0.01, "pass")}])

        with self.assertRaisesRegex(ValueError, "window"):
            analyze_policy_iteration_trend(run_dirs, window=0)

    def test_cli_exit_on_regression_returns_non_zero(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (0.10, "pass")},
                {"candidate": (0.05, "pass")},
                {"candidate": (-0.01, "fail")},
            ]
        )

        result = CLI_RUNNER.invoke(
            app,
            [
                "evaluate",
                "trend",
                *(str(run_dir) for run_dir in run_dirs),
                "--exit-on-regression",
            ],
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("degrading", result.stdout)

    def test_cli_json_out_writes_machine_readable_report(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (0.02, "pass")},
                {"candidate": (0.01, "pass")},
            ]
        )
        json_out = self.root / "trend-report.json"

        result = CLI_RUNNER.invoke(
            app,
            [
                "evaluate",
                "trend",
                *(str(run_dir) for run_dir in run_dirs),
                "--json-out",
                str(json_out),
            ],
        )

        self.assertEqual(result.exit_code, 0, result.stdout)
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(payload["candidate_variant_id"], "candidate")
        self.assertEqual(payload["split"], "holdout")
        self.assertEqual(payload["iterations_analyzed"], 2)
        self.assertEqual(payload["mean_utility_trend"]["direction"], "degrading")
        self.assertEqual(len(payload["iteration_points"]), 2)

    def _write_sequence(
        self,
        runs: Sequence[Mapping[str, tuple[float, str]]],
        *,
        missing_scorecard_indices: Iterable[int] = (),
        missing_decisions_indices: Iterable[int] = (),
        omit_holdout_mean_utility_indices: Iterable[int] = (),
    ) -> list[Path]:
        missing_scorecard = set(missing_scorecard_indices)
        missing_decisions = set(missing_decisions_indices)
        omit_holdout_mean_utility = set(omit_holdout_mean_utility_indices)

        run_dirs: list[Path] = []
        for index, candidate_map in enumerate(runs, start=1):
            run_dir = self.root / f"run-{index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if index - 1 not in missing_scorecard:
                self._write_scorecard(
                    run_dir / "scorecard.csv",
                    candidate_map,
                    omit_holdout_mean_utility=index - 1 in omit_holdout_mean_utility,
                )
            if index - 1 not in missing_decisions:
                self._write_promotion_decisions(run_dir / "promotion_decisions.json", candidate_map)
            run_dirs.append(run_dir)
        return run_dirs

    def _write_scorecard(
        self,
        path: Path,
        candidates: Mapping[str, tuple[float, str]],
        *,
        omit_holdout_mean_utility: bool = False,
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
                        delta=delta,
                    )
                )
            rows.append(
                self._scorecard_row(
                    split="dev",
                    candidate_variant_id=candidate_variant_id,
                    metric="mean_utility",
                    metric_group="utility",
                    delta=delta + 1.0,
                )
            )
            rows.append(
                self._scorecard_row(
                    split="holdout",
                    candidate_variant_id=candidate_variant_id,
                    metric="median_utility",
                    metric_group="utility",
                    delta=delta,
                )
            )

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCORECARD_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def _write_promotion_decisions(self, path: Path, candidates: Mapping[str, tuple[float, str]]) -> None:
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
        delta: float,
    ) -> dict[str, object]:
        baseline_value = 0.8
        candidate_value = baseline_value + delta
        return {
            "profile_id": "replay_outcomes_v1",
            "split": split,
            "baseline_variant_id": "baseline",
            "candidate_variant_id": candidate_variant_id,
            "metric": metric,
            "metric_group": metric_group,
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
            "delta": delta,
            "ci_low": "",
            "ci_high": "",
            "n_cases": 8,
            "n_paired": 8,
            "paired_coverage_rate": 1.0,
        }
