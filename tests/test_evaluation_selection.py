from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from policy_eval_harness.evaluation import run_evaluation_from_manifest
from tests.support.evaluation_fixtures import EvaluationFixtureMixin


class SelectionEvaluationTests(EvaluationFixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

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

    def test_selection_panel_requires_paired_holdout_coverage_for_verdict(self) -> None:
        manifest_path = self._write_selection_manifest(
            panel_rows=[
                {"case_id": "case-a", "variant_id": "baseline", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
                {"case_id": "case-b", "variant_id": "candidate", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
                {"case_id": "case-c", "variant_id": "candidate", "split": "holdout", "selected": False, "label": False, "utility": 0.0},
            ],
        )

        artifacts = run_evaluation_from_manifest(manifest_path, self.root / "selection-no-pair")
        scorecard = pd.read_csv(artifacts.scorecard_path)
        decisions = json.loads(artifacts.promotion_decisions_path.read_text(encoding="utf-8"))

        balanced_accuracy = scorecard[
            (scorecard["candidate_variant_id"] == "candidate") & (scorecard["metric"] == "balanced_accuracy")
        ].iloc[0]
        self.assertEqual(int(balanced_accuracy["n_paired"]), 0)
        self.assertTrue(pd.isna(balanced_accuracy["candidate_value"]))
        self.assertEqual(decisions["candidates"][0]["verdict"], "no_verdict")
        self.assertEqual(decisions["candidates"][0]["failure_reasons"], ["missing paired holdout comparison"])

    def test_selection_panel_fails_partial_paired_coverage_by_default(self) -> None:
        panel_rows = [
            {"case_id": "case-a", "variant_id": "baseline", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
            {"case_id": "case-b", "variant_id": "baseline", "split": "holdout", "selected": False, "label": False, "utility": 0.0},
            {"case_id": "case-a", "variant_id": "candidate", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
        ]

        default_manifest = self._write_selection_manifest(
            panel_rows=panel_rows,
            gates_thresholds={
                "min_delta_balanced_accuracy_vs_chance": 0.0,
                "min_delta_mean_utility_vs_accept_all": 0.0,
                "min_delta_mean_utility_vs_random_rate_matched": 0.0,
            },
        )
        default_artifacts = run_evaluation_from_manifest(default_manifest, self.root / "partial-default")
        default_decisions = json.loads(default_artifacts.promotion_decisions_path.read_text(encoding="utf-8"))

        self.assertEqual(default_decisions["candidates"][0]["verdict"], "fail")
        self.assertIn("min_paired_coverage_rate", default_decisions["candidates"][0]["failure_reasons"])
        self.assertAlmostEqual(default_decisions["candidates"][0]["observed"]["paired_coverage_rate"], 0.5)
        self.assertAlmostEqual(default_decisions["candidates"][0]["thresholds"]["min_paired_coverage_rate"], 1.0)

        relaxed_manifest = self._write_selection_manifest(
            panel_rows=panel_rows,
            gates_thresholds={
                "min_delta_balanced_accuracy_vs_chance": 0.0,
                "min_delta_mean_utility_vs_accept_all": 0.0,
                "min_delta_mean_utility_vs_random_rate_matched": 0.0,
                "min_paired_coverage_rate": 0.5,
            },
            base_name="selection-relaxed",
        )
        relaxed_artifacts = run_evaluation_from_manifest(relaxed_manifest, self.root / "partial-relaxed")
        relaxed_decisions = json.loads(relaxed_artifacts.promotion_decisions_path.read_text(encoding="utf-8"))
        self.assertEqual(relaxed_decisions["candidates"][0]["verdict"], "pass")

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

        assert_frame_equal(pd.read_csv(first.scorecard_path), pd.read_csv(second.scorecard_path))
        first_panel = pd.read_parquet(first.comparison_panel_path)
        self.assertEqual(sorted(first_panel["split"].unique().tolist()), ["dev", "holdout"])

    def test_time_holdout_overrides_existing_split_and_existing_mode_preserves_it(self) -> None:
        panel_rows = [
            {"case_id": "case-old", "variant_id": "baseline", "split": "holdout", "opened_at_utc": "2026-01-01T00:00:00Z", "selected": False, "label": False, "utility": 0.0},
            {"case_id": "case-old", "variant_id": "candidate", "split": "holdout", "opened_at_utc": "2026-01-01T00:00:00Z", "selected": False, "label": False, "utility": 0.0},
            {"case_id": "case-new", "variant_id": "baseline", "split": "dev", "opened_at_utc": "2026-07-01T00:00:00Z", "selected": True, "label": True, "utility": 1.0},
            {"case_id": "case-new", "variant_id": "candidate", "split": "dev", "opened_at_utc": "2026-07-01T00:00:00Z", "selected": True, "label": True, "utility": 1.0},
        ]

        time_manifest = self._write_selection_manifest(
            panel_rows=panel_rows,
            splits={
                "mode": "time_holdout",
                "field": "opened_at_utc",
                "holdout_cutoff_utc": "2026-06-01T00:00:00Z",
            },
        )
        time_artifacts = run_evaluation_from_manifest(time_manifest, self.root / "time-override")
        time_panel = pd.read_parquet(time_artifacts.comparison_panel_path)
        time_splits = time_panel.drop_duplicates("case_id").set_index("case_id")["split"].to_dict()
        self.assertEqual(time_splits, {"case-old": "dev", "case-new": "holdout"})

        existing_manifest = self._write_selection_manifest(
            panel_rows=panel_rows,
            splits={"mode": "existing"},
            base_name="selection-existing",
        )
        existing_artifacts = run_evaluation_from_manifest(existing_manifest, self.root / "existing-split")
        existing_panel = pd.read_parquet(existing_artifacts.comparison_panel_path)
        existing_splits = existing_panel.drop_duplicates("case_id").set_index("case_id")["split"].to_dict()
        self.assertEqual(existing_splits, {"case-old": "holdout", "case-new": "dev"})

    def test_selection_panel_baselines_are_stable_across_equivalent_manifest_locations(self) -> None:
        panel_rows = [
            {"case_id": "case-a", "variant_id": "baseline", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
            {"case_id": "case-b", "variant_id": "baseline", "split": "holdout", "selected": True, "label": False, "utility": -0.5},
            {"case_id": "case-c", "variant_id": "baseline", "split": "holdout", "selected": False, "label": True, "utility": 1.2},
            {"case_id": "case-d", "variant_id": "baseline", "split": "holdout", "selected": False, "label": False, "utility": -0.3},
            {"case_id": "case-a", "variant_id": "candidate", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
            {"case_id": "case-b", "variant_id": "candidate", "split": "holdout", "selected": False, "label": False, "utility": -0.5},
            {"case_id": "case-c", "variant_id": "candidate", "split": "holdout", "selected": True, "label": True, "utility": 1.2},
            {"case_id": "case-d", "variant_id": "candidate", "split": "holdout", "selected": False, "label": False, "utility": -0.3},
        ]
        gates_thresholds = {
            "min_delta_balanced_accuracy_vs_chance": 0.1,
            "min_delta_mean_utility_vs_accept_all": 0.0,
            "min_delta_mean_utility_vs_random_rate_matched": 0.2,
            "min_accept_rate": 0.25,
            "max_accept_rate": 0.75,
            "max_parse_fail_rate": 0.0,
            "max_invalid_output_rate": 0.0,
            "max_error_decision_rate": 0.0,
        }

        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first_root = Path(first_tmp)
            second_root = Path(second_tmp)
            first_manifest = self._write_selection_manifest(
                panel_rows=panel_rows,
                gates_thresholds=gates_thresholds,
                root=first_root,
            )
            second_manifest = self._write_selection_manifest(
                panel_rows=panel_rows,
                gates_thresholds=gates_thresholds,
                root=second_root,
            )

            first = run_evaluation_from_manifest(first_manifest, first_root / "out")
            second = run_evaluation_from_manifest(second_manifest, second_root / "out")

            assert_frame_equal(pd.read_csv(first.scorecard_path), pd.read_csv(second.scorecard_path))
            self.assertEqual(
                json.loads(first.promotion_decisions_path.read_text(encoding="utf-8")),
                json.loads(second.promotion_decisions_path.read_text(encoding="utf-8")),
            )

    def test_selection_panel_rejects_duplicate_case_rows(self) -> None:
        manifest_path = self._write_selection_manifest(
            panel_rows=[
                {"case_id": "case-a", "variant_id": "baseline", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
                {"case_id": "case-a", "variant_id": "baseline", "split": "holdout", "selected": False, "label": False, "utility": 0.0},
                {"case_id": "case-a", "variant_id": "candidate", "split": "holdout", "selected": True, "label": True, "utility": 1.0},
            ],
        )

        with self.assertRaisesRegex(ValueError, "Duplicate evaluation rows"):
            run_evaluation_from_manifest(manifest_path, self.root / "duplicate-selection")

