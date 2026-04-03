from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from policy_eval_harness.cli import app
from policy_eval_harness.workflows import (
    run_ablation_2x2_from_manifest,
    run_label_compare_from_manifest,
)

CLI_RUNNER = CliRunner()


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.repo_root = Path(__file__).resolve().parents[1]
        self.label_root = self.repo_root / "examples" / "label_compare"
        self.ablation_root = self.repo_root / "examples" / "ablation_2x2"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_label_compare_run_is_deterministic_and_clean_label_wins(self) -> None:
        manifest = self.label_root / "label_compare.yaml"
        first = run_label_compare_from_manifest(manifest, self.root / "first")
        second = run_label_compare_from_manifest(manifest, self.root / "second")

        first_auc = pd.read_csv(first.oos_auc_path)
        second_auc = pd.read_csv(second.oos_auc_path)
        assert_frame_equal(first_auc, second_auc, check_dtype=False)

        for model_family in first_auc["model_family"].unique().tolist():
            subset = first_auc[first_auc["model_family"] == model_family].set_index("label_variant")
            self.assertGreater(subset.loc["clean_label", "oos_auc"], subset.loc["baseline_label", "oos_auc"])
            self.assertEqual(len(set(subset["train_id_hash"].tolist())), 1)
            self.assertEqual(len(set(subset["oos_id_hash"].tolist())), 1)

    def test_label_compare_cli_matches_golden_outputs(self) -> None:
        out_dir = self.root / "label-cli"
        manifest = self.label_root / "label_compare.yaml"

        result = CLI_RUNNER.invoke(
            app,
            ["workflow", "label-compare", "run", "--manifest", str(manifest), "--out-dir", str(out_dir)],
        )

        self.assertEqual(result.exit_code, 0, result.stdout)
        expected_files = [
            Path("oos_auc_by_label.csv"),
            Path("model_label_summary.json"),
            Path("feature_manifest_snapshot.json"),
        ]
        for relative_path in expected_files:
            self._assert_file_matches(
                out_dir / relative_path,
                self.label_root / "golden" / relative_path,
            )

    def test_ablation_run_reports_negative_interaction(self) -> None:
        manifest = self.ablation_root / "ablation_2x2.yaml"
        artifacts = run_ablation_2x2_from_manifest(manifest, self.root / "ablation")

        factor_effects = pd.read_csv(artifacts.factor_effects_path).set_index("metric")
        interactions = pd.read_csv(artifacts.interaction_summary_path).set_index("metric")

        self.assertAlmostEqual(factor_effects.loc["mean_utility", "a_effect"], 0.155, places=6)
        self.assertAlmostEqual(factor_effects.loc["mean_utility", "b_effect"], 0.0875, places=6)
        self.assertAlmostEqual(factor_effects.loc["mean_utility", "combined_effect"], 0.095, places=6)
        self.assertAlmostEqual(factor_effects.loc["mean_utility", "interaction_term"], -0.1475, places=6)
        self.assertEqual(interactions.loc["mean_utility", "qualitative_read"], "interference")

    def test_ablation_cli_matches_golden_outputs(self) -> None:
        out_dir = self.root / "ablation-cli"
        manifest = self.ablation_root / "ablation_2x2.yaml"

        result = CLI_RUNNER.invoke(
            app,
            ["workflow", "ablation-2x2", "run", "--manifest", str(manifest), "--out-dir", str(out_dir)],
        )

        self.assertEqual(result.exit_code, 0, result.stdout)
        expected_files = [
            Path("factor_effects.csv"),
            Path("interaction_summary.csv"),
            Path("ablation_report.md"),
        ]
        for relative_path in expected_files:
            self._assert_file_matches(
                out_dir / relative_path,
                self.ablation_root / "golden" / relative_path,
            )

    def _assert_file_matches(self, actual_path: Path, expected_path: Path) -> None:
        self.assertTrue(actual_path.exists(), str(actual_path))
        self.assertTrue(expected_path.exists(), str(expected_path))
        suffix = actual_path.suffix.lower()
        if suffix == ".json":
            self.assertEqual(
                json.loads(actual_path.read_text(encoding="utf-8")),
                json.loads(expected_path.read_text(encoding="utf-8")),
            )
            return
        if suffix == ".csv":
            assert_frame_equal(
                self._normalized_frame(pd.read_csv(actual_path)),
                self._normalized_frame(pd.read_csv(expected_path)),
                check_dtype=False,
            )
            return
        if suffix == ".md":
            self.assertEqual(actual_path.read_text(encoding="utf-8"), expected_path.read_text(encoding="utf-8"))
            return
        self.fail(f"Unsupported artifact type for comparison: {actual_path}")

    def _normalized_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.reset_index(drop=True)
        return frame.sort_values(list(frame.columns)).reset_index(drop=True)
