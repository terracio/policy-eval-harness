from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from policy_eval_harness.cli import app
from policy_eval_harness.evaluation import run_evaluation_from_manifest

CLI_RUNNER = CliRunner()


class EvaluationWorkflowTests(unittest.TestCase):
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

    def test_selection_panel_profile_computes_metrics_and_gate_baselines(self) -> None:
        manifest_path = self._write_selection_manifest(
            panel_rows=[
                {"case_id": "case-a", "variant_id": "baseline", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
                {"case_id": "case-b", "variant_id": "baseline", "split": "holdout", "selected": True, "label": False, "utility": -0.5},
                {"case_id": "case-c", "variant_id": "baseline", "split": "holdout", "selected": False, "label": True, "utility": 1.2},
                {"case_id": "case-d", "variant_id": "baseline", "split": "holdout", "selected": False, "label": False, "utility": -0.3},
                {"case_id": "case-a", "variant_id": "candidate", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
                {"case_id": "case-b", "variant_id": "candidate", "split": "holdout", "selected": False, "label": False, "utility": -0.5},
                {"case_id": "case-c", "variant_id": "candidate", "split": "holdout", "selected": True, "label": True, "utility": 1.2},
                {"case_id": "case-d", "variant_id": "candidate", "split": "holdout", "selected": False, "label": False, "utility": -0.3},
            ],
            gates_thresholds={
                "min_delta_balanced_accuracy_vs_chance": 0.1,
                "min_delta_mean_utility_vs_accept_all": 0.0,
                "min_delta_mean_utility_vs_random_rate_matched": 0.2,
                "min_accept_rate": 0.25,
                "max_accept_rate": 0.75,
                "max_parse_fail_rate": 0.0,
                "max_invalid_output_rate": 0.0,
                "max_error_decision_rate": 0.0,
            },
        )

        artifacts = run_evaluation_from_manifest(manifest_path, self.root / "selection")
        scorecard = pd.read_csv(artifacts.scorecard_path)
        decisions = json.loads(artifacts.promotion_decisions_path.read_text(encoding="utf-8"))

        balanced_accuracy = scorecard[
            (scorecard["candidate_variant_id"] == "candidate") & (scorecard["metric"] == "balanced_accuracy")
        ].iloc[0]
        accept_rate = scorecard[
            (scorecard["candidate_variant_id"] == "candidate") & (scorecard["metric"] == "accept_rate")
        ].iloc[0]
        self.assertAlmostEqual(balanced_accuracy["candidate_value"], 1.0)
        self.assertAlmostEqual(accept_rate["candidate_value"], 0.5)
        self.assertEqual(decisions["candidates"][0]["verdict"], "pass")

    def test_time_holdout_and_bootstrap_are_deterministic(self) -> None:
        manifest_path = self._write_selection_manifest(
            panel_rows=[
                {"case_id": "case-a", "variant_id": "baseline", "opened_at_utc": "2026-01-01T00:00:00Z", "selected": False, "label": False, "utility": 0.1},
                {"case_id": "case-b", "variant_id": "baseline", "opened_at_utc": "2026-01-02T00:00:00Z", "selected": True, "label": True, "utility": 1.0},
                {"case_id": "case-c", "variant_id": "baseline", "opened_at_utc": "2026-01-03T00:00:00Z", "selected": True, "label": False, "utility": 0.2},
                {"case_id": "case-d", "variant_id": "baseline", "opened_at_utc": "2026-01-04T00:00:00Z", "selected": False, "label": True, "utility": 0.8},
                {"case_id": "case-a", "variant_id": "candidate", "opened_at_utc": "2026-01-01T00:00:00Z", "selected": False, "label": False, "utility": 0.1},
                {"case_id": "case-b", "variant_id": "candidate", "opened_at_utc": "2026-01-02T00:00:00Z", "selected": True, "label": True, "utility": 1.0},
                {"case_id": "case-c", "variant_id": "candidate", "opened_at_utc": "2026-01-03T00:00:00Z", "selected": False, "label": False, "utility": 0.2},
                {"case_id": "case-d", "variant_id": "candidate", "opened_at_utc": "2026-01-04T00:00:00Z", "selected": True, "label": True, "utility": 0.8},
            ],
            include_split=False,
            bootstrap={"enabled": True, "n_samples": 25, "seed": 13},
            splits={"mode": "time_holdout", "field": "opened_at_utc", "holdout_pct": 0.5},
            gates_thresholds={"min_delta_balanced_accuracy_vs_chance": 0.0},
        )

        first = run_evaluation_from_manifest(manifest_path, self.root / "first")
        second = run_evaluation_from_manifest(manifest_path, self.root / "second")

        first_scorecard = pd.read_csv(first.scorecard_path)
        second_scorecard = pd.read_csv(second.scorecard_path)
        assert_frame_equal(first_scorecard, second_scorecard)
        first_panel = pd.read_parquet(first.comparison_panel_path)
        self.assertEqual(sorted(first_panel["split"].unique().tolist()), ["dev", "holdout"])

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

    def _write_replay_manifest(
        self,
        *,
        summary_rows: List[Dict[str, Any]],
        cases_rows: List[Dict[str, Any]],
        gates_thresholds: Dict[str, Any] | None = None,
    ) -> Path:
        summary_path = self.root / "episode_summary.csv"
        cases_path = self.root / "cases.csv"
        manifest_path = self.root / "replay-evaluate.yaml"

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        pd.DataFrame(cases_rows).to_csv(cases_path, index=False)

        manifest = {
            "profile": "replay_outcomes_v1",
            "inputs": {
                "episode_summary_path": summary_path.name,
                "cases_path": cases_path.name,
            },
            "baseline_variant_id": "baseline",
            "candidate_variant_ids": ["candidate"],
            "gates": {
                "profile": "sequential_promotion_v1",
                "thresholds": gates_thresholds or {},
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _write_selection_manifest(
        self,
        *,
        panel_rows: List[Dict[str, Any]],
        gates_thresholds: Dict[str, Any] | None = None,
        bootstrap: Dict[str, Any] | None = None,
        splits: Dict[str, Any] | None = None,
        include_split: bool = True,
    ) -> Path:
        panel_path = self.root / "panel.csv"
        manifest_path = self.root / "selection-evaluate.yaml"

        frame = pd.DataFrame(panel_rows)
        if not include_split and "split" in frame.columns:
            frame = frame.drop(columns=["split"])
        frame.to_csv(panel_path, index=False)

        manifest = {
            "profile": "selection_panel_v1",
            "inputs": {
                "panel_path": panel_path.name,
            },
            "baseline_variant_id": "baseline",
            "candidate_variant_ids": ["candidate"],
            "splits": splits or {"mode": "existing"},
            "gates": {
                "profile": "selection_promotion_v1",
                "thresholds": gates_thresholds or {},
            },
        }
        if bootstrap is not None:
            manifest["bootstrap"] = bootstrap
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def _summary_row(
        self,
        case_id: str,
        variant_id: str,
        utility: float | None,
        *,
        decisions: int = 1,
        terminated: bool = False,
        final_status: str = "completed",
        error_count: int = 0,
        termination_reason: str | None = None,
    ) -> Dict[str, Any]:
        summary_metrics = {} if utility is None else {"utility": utility}
        return {
            "case_id": case_id,
            "variant_id": variant_id,
            "start_ts_utc": "2026-01-01T00:00:00Z",
            "end_ts_utc": "2026-01-01T00:05:00Z",
            "n_steps": 1,
            "terminated": terminated,
            "termination_reason": termination_reason,
            "final_status": final_status,
            "decision_count": decisions,
            "error_count": error_count,
            "summary_metrics_json": json.dumps(summary_metrics, sort_keys=True),
        }
