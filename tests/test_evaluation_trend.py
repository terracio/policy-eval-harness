from __future__ import annotations

import math
import unittest

from policy_eval_harness.evaluation import analyze_policy_iteration_trend
from tests.support.evaluation_trend_fixtures import EvaluationTrendFixtureMixin


class EvaluationTrendTests(EvaluationTrendFixtureMixin, unittest.TestCase):
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

    def test_no_verdict_blank_delta_is_accepted_and_excluded_from_regression_math(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (0.06, "pass")},
                {"candidate": (None, "no_verdict")},
                {"candidate": (0.03, "pass")},
            ],
            blank_delta_indices={1},
        )

        report = analyze_policy_iteration_trend(run_dirs)

        self.assertEqual(report.iterations_analyzed, 3)
        self.assertEqual(report.pass_count, 2)
        self.assertEqual(report.no_verdict_count, 1)
        self.assertAlmostEqual(report.promotion_rate, 2 / 3)
        self.assertEqual(report.mean_utility_trend.direction, "degrading")
        self.assertAlmostEqual(report.mean_utility_trend.first_value, 0.06)
        self.assertAlmostEqual(report.mean_utility_trend.last_value, 0.03)
        self.assertEqual(report.consecutive_decline_streak, 0)
        self.assertTrue(math.isnan(report.iteration_points[1].mean_utility_delta))

    def test_returns_stable_report_when_fewer_than_two_finite_deltas_remain(self) -> None:
        run_dirs = self._write_sequence(
            [
                {"candidate": (None, "no_verdict")},
                {"candidate": (0.03, "pass")},
            ],
            blank_delta_indices={0},
        )

        report = analyze_policy_iteration_trend(run_dirs)

        self.assertEqual(report.iterations_analyzed, 2)
        self.assertEqual(report.mean_utility_trend.direction, "stable")
        self.assertAlmostEqual(report.mean_utility_trend.slope, 0.0)
        self.assertFalse(report.any_regression)
        self.assertAlmostEqual(report.mean_utility_trend.first_value, 0.03)
        self.assertAlmostEqual(report.mean_utility_trend.last_value, 0.03)
        self.assertEqual(report.consecutive_decline_streak, 0)

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
