from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from policy_eval_harness.cli import app
from policy_eval_harness.evaluation import run_evaluation_from_manifest
from tests.support.evaluation_fixtures import EvaluationFixtureMixin

CLI_RUNNER = CliRunner()


class ReplayEvaluationTests(EvaluationFixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_replay_outcomes_profile_builds_panel_and_scorecard(self) -> None:
        manifest_path = self._write_replay_manifest(
            summary_rows=[
                self._summary_row("case-a", "baseline", 0.1, decisions=1),
                self._summary_row("case-b", "baseline", 0.2, decisions=2),
                self._summary_row("case-a", "candidate", 0.4, decisions=1),
                self._summary_row("case-b", "candidate", 0.5, decisions=1),
            ],
            cases_rows=[
                {"case_id": "case-a", "split": "dev"},
                {"case_id": "case-b", "split": "holdout"},
            ],
            gates_thresholds={"min_delta_mean_utility": 0.2},
        )

        artifacts = run_evaluation_from_manifest(manifest_path, self.root / "out")

        panel = pd.read_parquet(artifacts.comparison_panel_path)
        scorecard = pd.read_csv(artifacts.scorecard_path)
        decisions = json.loads(artifacts.promotion_decisions_path.read_text(encoding="utf-8"))

        self.assertEqual(sorted(panel["split"].unique().tolist()), ["dev", "holdout"])
        holdout_mean = scorecard[
            (scorecard["split"] == "holdout")
            & (scorecard["candidate_variant_id"] == "candidate")
            & (scorecard["metric"] == "mean_utility")
        ].iloc[0]
        self.assertAlmostEqual(holdout_mean["baseline_value"], 0.2)
        self.assertAlmostEqual(holdout_mean["candidate_value"], 0.5)
        self.assertAlmostEqual(holdout_mean["delta"], 0.3)
        self.assertEqual(decisions["candidates"][0]["verdict"], "pass")

    def test_replay_profile_reports_pairing_coverage_and_no_verdict_when_utility_missing(self) -> None:
        coverage_manifest = self._write_replay_manifest(
            summary_rows=[
                self._summary_row("case-a", "baseline", 1.0),
                self._summary_row("case-b", "baseline", 1.0),
                self._summary_row("case-a", "candidate", 2.0),
            ],
            cases_rows=[
                {"case_id": "case-a", "split": "holdout"},
                {"case_id": "case-b", "split": "holdout"},
            ],
            gates_thresholds={"min_paired_coverage_rate": 0.5},
        )
        coverage_artifacts = run_evaluation_from_manifest(coverage_manifest, self.root / "coverage")
        coverage_scorecard = pd.read_csv(coverage_artifacts.scorecard_path)
        coverage_row = coverage_scorecard[
            (coverage_scorecard["split"] == "holdout")
            & (coverage_scorecard["candidate_variant_id"] == "candidate")
            & (coverage_scorecard["metric"] == "paired_coverage_rate")
        ].iloc[0]
        self.assertAlmostEqual(coverage_row["candidate_value"], 0.5)
        self.assertEqual(int(coverage_row["n_cases"]), 2)
        self.assertEqual(int(coverage_row["n_paired"]), 1)

        no_utility_manifest = self._write_replay_manifest(
            summary_rows=[
                self._summary_row("case-a", "baseline", None),
                self._summary_row("case-a", "candidate", None),
            ],
            cases_rows=[{"case_id": "case-a", "split": "holdout"}],
        )
        no_utility_artifacts = run_evaluation_from_manifest(no_utility_manifest, self.root / "no-utility")
        no_utility_decisions = json.loads(
            no_utility_artifacts.promotion_decisions_path.read_text(encoding="utf-8")
        )
        self.assertEqual(no_utility_decisions["candidates"][0]["verdict"], "no_verdict")

    def test_cli_evaluate_run_writes_artifacts(self) -> None:
        manifest_path = self._write_replay_manifest(
            summary_rows=[
                self._summary_row("case-a", "baseline", 0.2),
                self._summary_row("case-a", "candidate", 0.5),
            ],
            cases_rows=[{"case_id": "case-a", "split": "holdout"}],
        )
        out_dir = self.root / "cli-out"

        result = CLI_RUNNER.invoke(
            app,
            ["evaluate", "run", "--manifest", str(manifest_path), "--out-dir", str(out_dir)],
        )

        self.assertEqual(result.exit_code, 0, result.stdout)
        self.assertIn("comparison_panel.parquet", result.stdout)
        self.assertTrue((out_dir / "comparison_panel.parquet").exists())
        self.assertTrue((out_dir / "scorecard.csv").exists())
        self.assertTrue((out_dir / "promotion_decisions.json").exists())

    def test_replay_profile_rejects_duplicate_case_rows(self) -> None:
        manifest_path = self._write_replay_manifest(
            summary_rows=[
                self._summary_row("case-a", "baseline", 0.1),
                self._summary_row("case-a", "baseline", 0.2),
                self._summary_row("case-a", "candidate", 0.3),
            ],
            cases_rows=[{"case_id": "case-a", "split": "holdout"}],
        )

        with self.assertRaisesRegex(ValueError, "Duplicate evaluation rows"):
            run_evaluation_from_manifest(manifest_path, self.root / "duplicate-replay")

