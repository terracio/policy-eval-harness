from __future__ import annotations

import json
import unittest

from typer.testing import CliRunner

from policy_eval_harness.cli import app
from policy_eval_harness.evaluation import analyze_policy_iteration_trend
from tests.support.evaluation_trend_fixtures import EvaluationTrendFixtureMixin

CLI_RUNNER = CliRunner()


class EvaluationTrendErrorTests(EvaluationTrendFixtureMixin, unittest.TestCase):
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

    def test_errors_when_non_no_verdict_row_has_blank_delta(self) -> None:
        run_dirs = self._write_sequence(
            [{"candidate": (None, "pass")}],
            blank_delta_indices={0},
        )

        with self.assertRaisesRegex(ValueError, "Scorecard delta is invalid"):
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

    def test_rejects_non_positive_window(self) -> None:
        run_dirs = self._write_sequence([{"candidate": (0.01, "pass")}])

        with self.assertRaisesRegex(ValueError, "window"):
            analyze_policy_iteration_trend(run_dirs, window=0)

    def test_errors_when_promotion_decisions_root_is_not_an_object(self) -> None:
        run_dirs = self._write_sequence(
            [{"candidate": (0.01, "pass")}],
            invalid_decisions_root_indices={0},
        )

        with self.assertRaisesRegex(ValueError, "JSON object"):
            analyze_policy_iteration_trend(run_dirs)

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

    def test_cli_reports_clean_error_for_invalid_promotion_decisions_payload(self) -> None:
        run_dirs = self._write_sequence(
            [{"candidate": (0.02, "pass")}],
            invalid_decisions_root_indices={0},
        )

        result = CLI_RUNNER.invoke(
            app,
            [
                "evaluate",
                "trend",
                *(str(run_dir) for run_dir in run_dirs),
            ],
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("promotion_decisions.json", result.stderr)
        self.assertNotIn("Traceback", result.output)
